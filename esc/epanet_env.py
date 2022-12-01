import os
import shutil
import tempfile

import numpy as np
from epyt import epanet
from gym import Env
from gym.spaces import Box, Discrete

from esc.electricity_rates import electricity_rate
from esc.reward import reward_high_tank_head, reward_low_energy_cost
from esc.water_usage import (
    TYPICAL_BUILDING_MEAN_WATER_CONSUMPTION_LITER_PER_MINUTE,
    relative_occupant_water_demand,
)
from esc.util import MEAN_WATER_TANK_HEIGHT_M, minute_in_day


SIMULATION_DURATION_S = 172800
SIMULATION_TIMESTEP_S = 60
N_SIMULATION_STEPS = SIMULATION_DURATION_S / SIMULATION_TIMESTEP_S

tankID = "T1"
pumpID = "PUMP"

# Number of simulation steps of the 
N = 10
M = 10

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
            low=np.array([  *np.zeros(N),        *(-np.inf * np.ones(M)),  -np.inf,                      0.0, 0.0]),
            high=np.array([ *(np.inf * np.ones(N)), *(np.inf * np.ones(M)), MEAN_WATER_TANK_HEIGHT_M, 1440.0, 1.0])
        )

    def reset(self):
        self.d = initialize_epanet()

        self.tank_index = self.d.getNodeIndex(tankID)
        self.pump_index = self.d.getLinkIndex(pumpID)
        self.tank_elevation = self.d.getNodeElevations(self.tank_index)

        self.tstep = 1
        self.i = 0
        self.pump_energy_cost: float = 0.0
        self.prev_action = 1

        H = self.d.getNodeHydraulicHead()
        tank_head = H[self.tank_index - 1] - self.tank_elevation

        return np.array(
            [
                *electricity_rate(np.arange(self.i - N, self.i)),  # type: ignore
                *relative_occupant_water_demand(np.arange(self.i - M, self.i)),  # type: ignore
                tank_head,
                minute_in_day(self.i),
                self.prev_action
            ]
        )

    def step(self, action):
        H = self.d.getNodeHydraulicHead()
        tank_head = H[self.tank_index - 1] - self.tank_elevation

        self.d.setLinkStatus(self.pump_index, action)
        self.i += 1

        self.d.runHydraulicAnalysis()

        pump_energy_usage = self.d.getLinkEnergy(self.pump_index)
        self.pump_energy_cost += electricity_rate(self.i) * pump_energy_usage  # type: ignore

        self.tstep = self.d.nextHydraulicAnalysisStep()

        info = {
            "pump_status": self.d.getLinkStatus(self.pump_index),
            "pump_energy_usage": pump_energy_usage,
            "flows": self.d.getLinkFlows(),
            "pressures": self.d.getNodePressure(),
        }

        obs = np.array(
                [
                    *electricity_rate(np.arange(self.i - N, self.i)),  # type: ignore
                    *relative_occupant_water_demand(np.arange(self.i - M, self.i)),  # type: ignore
                    np.round(tank_head, decimals=4),
                    minute_in_day(self.i),
                    self.prev_action,
                ]
            )

        reward = reward_low_energy_cost(self.pump_energy_cost) \
            * reward_high_tank_head(tank_head) \
            * (float(self.prev_action == action) + 1) / 2.0

        self.prev_action = action

        done = self.tstep <= 0
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
            info,
        )
