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
from esc.util import MEAN_WATER_TANK_HEIGHT_M


SIMULATION_DURATION_S = 172800
SIMULATION_TIMESTEP_S = 60
N_SIMULATION_STEPS = SIMULATION_DURATION_S / SIMULATION_TIMESTEP_S

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
    def __init__(self, env_config):
        # Action space is one of {0, 1}, where 0 means the pump is off, and 1
        # means the pump is on
        self.action_space = Discrete(2)
        # Observation space is 3 dimensional: 1st dimension is the current
        # electricity price, 2nd dimension is the current occupant water demand.
        # Electricity price is unbounded in either direction, but water demand
        # must be non-negative. Highest possible value is
        # MEAN_WATER_TANK_HEIGHT_M.
        self.observation_space = Box(
            low=np.array([0.0, -np.inf, 0.0]), high=np.array([np.inf, np.inf, MEAN_WATER_TANK_HEIGHT_M])
        )

    def reset(self):
        self.d = initialize_epanet()

        self.tank_index = self.d.getNodeIndex(tankID)
        self.pump_index = self.d.getLinkIndex(pumpID)
        self.tank_elevation = self.d.getNodeElevations(self.tank_index)

        self.tstep = 1
        self.i = 0
        self.pump_energy_cost: float = 0.0

        H = self.d.getNodeHydraulicHead()
        tank_head = H[self.tank_index - 1] - self.tank_elevation

        return np.array(
            [electricity_rate(self.i), relative_occupant_water_demand(self.i), tank_head]
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

        done = self.tstep <= 0
        if done:
            self.d.closeHydraulicAnalysis()
            self.d.unload()
            print("Hydraulic Analysis completed succesfully.")

        return (
            # Observation
            np.array(
                [electricity_rate(self.i), relative_occupant_water_demand(self.i), tank_head]
            ),
            # Reward
            reward_low_energy_cost(self.pump_energy_cost)
            * reward_high_tank_head(tank_head),
            # Whether or not the simulation is done
            done,
            # Diagnostic info
            info,
        )
