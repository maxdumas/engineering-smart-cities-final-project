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
    relative_occupant_water_demand,
)
from esc.util import MEAN_WATER_TANK_HEIGHT_M, MINUTES_PER_DAY


SIMULATION_DURATION_S = 172800
SIMULATION_TIMESTEP_S = 60
N_SIMULATION_STEPS = SIMULATION_DURATION_S / SIMULATION_TIMESTEP_S

MIN_ENERGY_PRICE = np.min(all_rates)
CHEAP_ENERGY_PRICE = np.quantile(all_rates, 0.4)
EXPENSIVE_ENERGY_PRICE = np.quantile(all_rates, 0.8)
MAX_ENERGY_PRICE = np.max(all_rates)

MIN_WATER_DEMAND = np.min(relative_occupant_water_demand(np.arange(MINUTES_PER_DAY)))
MAX_WATER_DEMAND = np.max(relative_occupant_water_demand(np.arange(MINUTES_PER_DAY)))

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
    
    # Simulation counters
    tstep: int
    i: int
    
    def __init__(self, env_config):
        # Action space is one of {0, 1}, where 0 means the pump is off, and 1
        # means the pump is on
        self.action_space = Discrete(2)
        # Observation space is N+M+2 dimensional: First N dimensions are the N
        # electricity prices in the simulation steps prior to the current step,
        # Dimensions N+1 through N+M are the N occupant water demand rates in
        # the simulation step prior to the current step. N+M+1 is the current
        # water tank height. N+M+2 is the minute of the day. Electricity price
        # is unbounded in either direction, but water demand must be
        # non-negative. Highest possible tank value is MEAN_WATER_TANK_HEIGHT_M.
        self.observation_space = Box(
            low=np.array([  MIN_ENERGY_PRICE, MIN_WATER_DEMAND,  0.0]),
            high=np.array([ MAX_ENERGY_PRICE, MAX_WATER_DEMAND,  MEAN_WATER_TANK_HEIGHT_M])
        )

    def reset(self):
        self.d = initialize_epanet()

        self.tank_index = self.d.getNodeIndex(tankID) # type: ignore
        self.pump_index = self.d.getLinkIndex(pumpID) # type: ignore
        self.tank_elevation = self.d.getNodeElevations(self.tank_index)  # type: ignore

        self.n_switches = 0

        self.tstep = 1
        self.i = 0

        # Initialize the system with the pump off.
        self.d.setLinkStatus(self.pump_index, 0)

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
                electricity_rate(self.i),
                relative_occupant_water_demand(self.i),
                self.get_tank_head(),
            ]
        )

    def step(self, action):
        # Gather observation data before action
        prev_pump_state = self.d.getLinkStatus(self.pump_index)
        
        # Perform action
        self.d.setLinkStatus(self.pump_index, action)

        # Update simulation state
        self.d.runHydraulicAnalysis()

        # Gather observation data after action
        curr_pump_state = self.d.getLinkStatus(self.pump_index)
        # pump_energy_usage = self.d.getLinkEnergy(self.pump_index)
        tank_head = self.get_tank_head()

        # Update accumulators
        energy_price = electricity_rate(self.i)
        # energy_cost = energy_price * pump_energy_usage
        # self.pump_energy_cost += energy_cost  # type: ignore

        # Compute observation and reward
        obs = self.observe()
        if prev_pump_state != curr_pump_state:
            # Strongly penalize any switching of the pump, so as to encourage
            # the pump to switch as little as possible.
            reward = -1.0
            self.n_switches += 1
        else:
            if tank_head > 1.5:
                if curr_pump_state == 0:
                    # If the pump is off, provide neutral reward.
                    reward = 0.0
                else:
                    # If the tank is above a minimum level and the pump is on,
                    # provide reward proportional to how much the current energy
                    # price is. For example, if the electricity price is $0.10
                    # kWh, provide -1.0 reward. If energy is free, provide 1.0
                    # reward.
                    reward = 1 - 2 * expit(6 * ((2 * energy_price - CHEAP_ENERGY_PRICE) / EXPENSIVE_ENERGY_PRICE - 1))
            else:
                # Strongly penalize ever allowing the tank to dip below minimum
                # allowed levels.
                reward = -1.0

        # Update simulation counters
        self.tstep = self.d.nextHydraulicAnalysisStep()
        self.i += 1

        # Check if we've completed the simulation. Sometimes the hydraulic
        # simulation wants to run a little longer than N_SIMULATION_STEPS so we
        # put a hard cap in addition to EPANET's own logic. We also have failed
        # if the tank is empty or if we've entered crazy toggling behavior.
        done = self.tstep <= 0 or self. i > N_SIMULATION_STEPS or tank_head <= 0.0 or self.n_switches > 50
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
