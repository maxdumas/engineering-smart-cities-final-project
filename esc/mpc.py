import do_mpc
import matplotlib.pyplot as plt
import numpy as np
from casadi import *
from casadi.tools import *

from esc.water_usage import typical_building_water_demand
from esc.electricity_rates import electricity_rate

N_SIMULATION_STEPS = 2880
N_HORIZON = 720
TANK_VOLUME_L = 38430
TYPICAL_BUILDING_HEIGHT_M = 34.75138

PUMP_MAX_FLOW_LPM = 150
PUMP_POWER_USAGE_W = 1200

TURBINE_OUTFLOW_LPM = 350
TURBINE_COMPONENT_EFFICIENCY = 0.85
G = 9.81
WATT_SEC_TO_KWH = 1.0 / 3600.0 / 1000.0
TURBINE_NET_HEAD = TYPICAL_BUILDING_HEIGHT_M + 1.5
HEAD_LOSS = 0.9

TURBINE_OUTFLOW_LPS = TURBINE_OUTFLOW_LPM / 60
TURBINE_POWER_GENERATION_W = 1.5 * TURBINE_OUTFLOW_LPS * G * TURBINE_NET_HEAD * HEAD_LOSS * TURBINE_COMPONENT_EFFICIENCY

def template_simulator(model):
    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step=1.0)

    tvp_template = simulator.get_tvp_template()

    def tvp_fun(t_now):
        tvp_template["electricity_rate"] = electricity_rate(t_now)
        tvp_template["water_demand"] = typical_building_water_demand(t_now)
        return tvp_template

    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    return simulator


def template_mpc(model):
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        "n_robust": 1,
        "n_horizon": N_HORIZON,
        "t_step": 1.0,
        "store_full_solution": True,
    }

    mpc.set_param(**setup_mpc)

    mterm = model.aux["cost"]
    lterm = model.aux["cost"]  # terminal cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1e-4)

    max_x = np.array([[TANK_VOLUME_L], [np.inf]])

    mpc.bounds["lower", "_x", "x"] = np.zeros_like(max_x)
    mpc.bounds["upper", "_x", "x"] = max_x

    mpc.bounds["lower", "_u", "u"] = 0.0
    mpc.bounds["upper", "_u", "u"] = 1.0

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        for k in range(N_HORIZON + 1):
            tvp_template["_tvp", k, "electricity_rate"] = electricity_rate(t_now + k)
            tvp_template["_tvp", k, "water_demand"] = typical_building_water_demand(t_now + k)
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    return mpc


def template_model(symvar_type="SX"):
    model_type = "discrete"  # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # States struct (optimization variables):
    # Our state is:
    # [
    #    tank volume,
    #    total energy cost
    #    potential energy cost, which is the most energy that could be spent if the pump was just always on
    # ]
    _x = model.set_variable(var_type="_x", var_name="x", shape=(2, 1))
    tank_vol = _x[0]
    energy_cost = _x[1]

    # Input struct (optimization variables):
    # 2-dimensional, first value controls pump inflow, second value controls turbine outflow
    _u = model.set_variable(var_type="_u", var_name="u", shape=(2, 1))
    pump_status = _u[0]
    turbine_status = _u[1]

    _electricity_rate = model.set_variable(var_type="_tvp", var_name="electricity_rate")
    _water_demand = model.set_variable(var_type="_tvp", var_name="water_demand")

    # Objective formulation
    tank_cost_component = 1 - 1 / (1 + exp(-6 * (2 * tank_vol / (TANK_VOLUME_L / 4) - 1)))
    energy_cost_component = 1 / (1 + exp(-6 * (2 * energy_cost / 1.25 - 1)))
    model.set_expression(expr_name="cost", expr=energy_cost_component + tank_cost_component)

    x_next = vertcat(
        tank_vol + PUMP_MAX_FLOW_LPM * pump_status - _water_demand - TURBINE_OUTFLOW_LPM * turbine_status,
        energy_cost + _electricity_rate * WATT_SEC_TO_KWH * 60 * (PUMP_POWER_USAGE_W * pump_status - TURBINE_POWER_GENERATION_W * turbine_status),
    )
    model.set_rhs("x", x_next)

    # Variables we want to easily observe in charts
    model.set_expression(expr_name="water_demand", expr=_water_demand)
    model.set_expression(expr_name="electricity_rate", expr=_electricity_rate)
    model.set_expression(expr_name="energy_cost", expr=energy_cost)

    model.setup()

    return model


def run():
    """ User settings: """
    show_animation = True
    store_results = False

    """
    Get configured do-mpc modules:
    """
    model = template_model()
    mpc = template_mpc(model)
    simulator = template_simulator(model)
    estimator = do_mpc.estimator.StateFeedback(model)


    """
    Set initial state
    """
    np.random.seed(99)

    x0 = np.array([TANK_VOLUME_L, 0.0])
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    """
    Setup graphic:
    """

    fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
    plt.ion()

    """
    Run MPC main loop:
    """

    u_total = []
    for k in range(N_SIMULATION_STEPS):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

        u_total.append(np.round(u0[0]))

        if show_animation:
            graphics.plot_results(t_ind=k)
            graphics.plot_predictions(t_ind=k)
            graphics.reset_axes()
            plt.show()
            plt.pause(0.01)

    
    np.save("pump_states.npy", np.array(u_total))
    input("Press any key to exit.")

    # Store results:
    if store_results:
        do_mpc.data.save_results([mpc, simulator], "water_tank")

    return model, mpc, simulator

if __name__ == "__main__":
    run()