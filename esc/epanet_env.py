import os
import shutil
import tempfile

import numpy as np
from epyt import epanet
from gym import Env
from gym.spaces import Box, Discrete
from scipy.special import expit

from esc.electricity_rates import electricity_rate, all_rates
from esc.reward import reward_high_tank_head, reward_low_energy_cost, reward_few_pump_switches
from esc.water_usage import (
    TYPICAL_BUILDING_MEAN_WATER_CONSUMPTION_LITER_PER_MINUTE,
    typical_building_water_demand,
    relative_occupant_water_demand
)
from esc.util import MEAN_WATER_TANK_HEIGHT_M, MINUTES_PER_DAY


SIMULATION_DURATION_S = 172800
SIMULATION_TIMESTEP_S = 60
N_SIMULATION_STEPS = SIMULATION_DURATION_S / SIMULATION_TIMESTEP_S

MIN_ENERGY_PRICE = np.min(all_rates)
CHEAP_ENERGY_PRICE = np.quantile(all_rates, 0.4)
EXPENSIVE_ENERGY_PRICE = np.quantile(all_rates, 0.8)
MAX_ENERGY_PRICE = np.max(all_rates)

MIN_WATER_DEMAND = np.min(typical_building_water_demand(np.arange(MINUTES_PER_DAY)))
MAX_WATER_DEMAND = np.max(typical_building_water_demand(np.arange(MINUTES_PER_DAY)))

tankID = "T1"
pumpID = "PUMP"

demand_sample_24h = relative_occupant_water_demand(np.arange(N_SIMULATION_STEPS))


def initialize_epanet():
    # EPANET steps on itself when multiple instances are loading the network
    # simulataneously. So each worker copies the original network file to a
    # temporary directory before opening it.
    with tempfile.TemporaryDirectory() as temp_dir:
        dest = os.path.join(temp_dir, "BUILDING.inp")
        shutil.copy("../networks/BUILDING.inp", dest)
        d = epanet(dest)
    d.setTimeSimulationDuration(SIMULATION_DURATION_S)  # 48 hour duration
    d.setTimeHydraulicStep(SIMULATION_TIMESTEP_S)  # Time step every minute
    d.setTimePatternStep(SIMULATION_TIMESTEP_S)  # Pattern step every minute

    # Add time-dependent pattern for occupant demand to the outflow junction
    # Sample the demand for every minute of the day
    d.addPattern("relative_occupant_demand", demand_sample_24h)
    d.setNodeJunctionData(
        1,
        0,
        TYPICAL_BUILDING_MEAN_WATER_CONSUMPTION_LITER_PER_MINUTE,
        "relative_occupant_demand",
    )

    # Hydraulic analysis STEP-BY-STEP.
    d.openHydraulicAnalysis()
    d.initializeHydraulicAnalysis(0)

    return d


class EPANETEnv(Env):
    d: epanet
    
    # Network constants
    tank_index: int
    pump_index: int
    tank_elevation: float

    n_switches: int
    pump_state: int
    
    # Simulation counters
    tstep: int
    i: int
    
    def __init__(self, env_config):
        # Action space is one of {0, 1}, where 0 means the pump is off, and 1
        # means the pump is on
        self.action_space = Discrete(2)

        self.observation_space = Box(
            low=np.array([  0.0, MIN_WATER_DEMAND,  0.0]),
            high=np.array([ 1.0, MAX_WATER_DEMAND,  MEAN_WATER_TANK_HEIGHT_M])
        )

    def reset(self):
        self.d = initialize_epanet()

        self.tank_index = self.d.getNodeIndex(tankID) # type: ignore
        self.pump_index = self.d.getLinkIndex(pumpID) # type: ignore
        self.tank_elevation = self.d.getNodeElevations(self.tank_index)  # type: ignore

        self.n_switches = 0
        self.pump_state = 0

        self.tstep = 1
        self.i = 0

        # Initialize the system with the pump off.
        self.d.setLinkStatus(self.pump_index, self.pump_state)

        return self.observe()
        
    def get_tank_head(self):
        h = self.d.getNodeHydraulicHead(self.tank_index) - self.tank_elevation
        if h <= 0.0:
            return 0.0
        else:
            return h
        
    def observe(self):
        return np.array(
            [
                self.pump_state,
                # electricity_rate(self.i),
                typical_building_water_demand(self.i),
                self.get_tank_head(),
            ]
        )

    def step(self, action):
        # Perform action
        if self.pump_state != action:
            # Only switch the pump if our current action represents a change
            # from the status quo.
            self.d.setLinkStatus(self.pump_index, action)
            self.pump_state = action  # type: ignore
            self.n_switches += 1

        # Update simulation state
        self.d.runHydraulicAnalysis()

        # pump_energy_usage = self.d.getLinkEnergy(self.pump_index)
        tank_head = self.get_tank_head()

        # Update accumulators
        # energy_price = electricity_rate(self.i)
        # energy_cost = energy_price * pump_energy_usage
        # self.pump_energy_cost += energy_cost  # type: ignore

        # Compute observation and reward
        obs = self.observe()

        if self.pump_state == 1:
            # If the pump is on, provide no reward. This represents us spending
            # money on electricity and is only desirable if necessary.
            reward = 0.0
        else:
            if tank_head > 1.5:
                # If the pump is off and the tank is full enough, provide
                # reward. This is what we want!
                reward = 1.0
            else:
                # If the tank is close to empty and the pump is off, provide
                # negative reward. This is because our other objective is to
                # ensure residents always have enough water.
                reward = -1.0

        # Update simulation counters
        self.tstep = self.d.nextHydraulicAnalysisStep()
        self.i += 1

        # Check if we've completed the simulation. Sometimes the hydraulic
        # simulation wants to run a little longer than N_SIMULATION_STEPS so we
        # put a hard cap in addition to EPANET's own logic. We also have failed
        # if the tank is empty.
        done = self.tstep <= 0 or self.i > N_SIMULATION_STEPS
        if done:
            self.d.closeHydraulicAnalysis()
            self.d.unload()
            print("Hydraulic Analysis completed succesfully.")

        return (
            # Observation
            obs,
            # Reward
            reward,
            # Whether or not the simulation is done
            done,
            # Diagnostic info
            {},
        )
