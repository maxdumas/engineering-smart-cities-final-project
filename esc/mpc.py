import do_mpc
import matplotlib.pyplot as plt
import numpy as np
from casadi import *
from casadi.tools import *

from esc.water_usage import typical_building_water_demand
from esc.electricity_rates import electricity_rate

N_SIMULATION_STEPS = 1440
N_HORIZON = 720
PUMP_MAX_FLOW_LPM = 150
TANK_VOLUME_L = 38430

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

    max_x = np.array([[TANK_VOLUME_L], [400.0]])

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

    # Simple oscillating masses example with two masses and two inputs.
    # States are the position and velocitiy of the two masses.

    # States struct (optimization variables):
    # Our state is:
    # [
    #    tank volume,
    #    total power cost
    # ]
    _x = model.set_variable(var_type="_x", var_name="x", shape=(2, 1))

    # Input struct (optimization variables):
    # This is meant to just be 0 or 1, to control whether the pump is on or not.
    _u = model.set_variable(var_type="_u", var_name="u", shape=(1, 1))

    _electricity_rate = model.set_variable(var_type="_tvp", var_name="electricity_rate")
    _water_demand = model.set_variable(var_type="_tvp", var_name="water_demand")

    # Observe electricity rate and water demand
    model.set_expression(expr_name="electricity_rate", expr=_electricity_rate)
    model.set_expression(expr_name="water_demand", expr=_water_demand)

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    tank_vol = _x[0]
    energy_cost = _x[1]
    model.set_expression(expr_name="power_cost", expr=energy_cost)
    tank_reward = 1 - 1 / (1 + exp(-6 * (2 * tank_vol / (TANK_VOLUME_L / 3) - 1)))
    energy_reward = 1 / (1 + exp(-6 * (2 * energy_cost / 50 - 1)))
    model.set_expression(expr_name="cost", expr=energy_reward + tank_reward)

    pump_status = _u[0] #if_else(_u[0] < 0.5, 0.0, 1.0)
    x_next = vertcat(
        _x[0] + PUMP_MAX_FLOW_LPM * pump_status - _water_demand,
        _x[1] + _electricity_rate * pump_status
    )
    model.set_rhs("x", x_next)

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

    x0 = np.array([TANK_VOLUME_L / 3, 0.0])
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

    for k in range(N_SIMULATION_STEPS):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

        if show_animation:
            graphics.plot_results(t_ind=k)
            graphics.plot_predictions(t_ind=k)
            graphics.reset_axes()
            plt.show()
            plt.pause(0.01)

    input("Press any key to exit.")

    # Store results:
    if store_results:
        do_mpc.data.save_results([mpc, simulator], "oscillating_masses")

    return model, mpc, simulator

if __name__ == "__main__":
    run()