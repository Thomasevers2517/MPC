import numpy as np
import control as ct
from functools import partial

import physical_modelling.initialisation as init
import animation
from physical_modelling.forward_kinematics import calculate_link_endpoints
from physical_modelling.dynamics import continuous_inverse_dynamics, continuous_state_space_dynamics
from physical_modelling.simulation import simulate_uncontrolled_system
from controllers import control_with_lqr_continuous, control_with_lqr_discrete
from plot_results import plot_errors, plot_trajectories

# simulation parameters
SIM_DT = 0.001    # s
SIM_DURATION = 1  # s

# initial conditions
Q_1_0 = 0                   # [rad]
PHI_0 = np.deg2rad(2)       # [rad]
THETA_0 = np.deg2rad(-2)    # [rad]
Q_2_0 = np.pi / 2           # [rad]
Q_1_DOT_0 = 0               # [rad]
PHI_DOT_0 = 0               # [rad]
THETA_DOT_0 = 0             # [rad]
Q_2_DOT_0 = 0               # [rad]

# control parameters
CTRL_DT = 0.001
Q = np.diag([7, 6, 6, 7, 0.0001, 0.0001, 0.0001, 0.0001])
R = np.diag([18, 18])


if __name__ == "__main__":
    physical_parameters = init.initialize_physical_parameters()
    t_ts_uncontrolled, sim_length_uncontrolled = init.generate_t_ts(SIM_DT, 0.65)  # singularity at t=0.65 s
    t_ts, sim_length = init.generate_t_ts(SIM_DT, SIM_DURATION)
    q_0, q_dot_0 = init.set_initial_conditions(Q_1_0, PHI_0, THETA_0, Q_2_0,
                                               Q_1_DOT_0, PHI_DOT_0, THETA_DOT_0, Q_2_DOT_0)

    link_ends = calculate_link_endpoints(physical_parameters, q_0)
    # animation.plot_mechanism_3d(link_ends)

    # couldn't get this to work
    # sim_ts = simulate_robot(
    #     physical_parameters=physical_parameters,
    #     t_ts=t_ts,
    #     discrete_forward_dynamics_fn=partial(linearized_discrete_forward_dynamics,
    #                                          Ad, Bd, Cd, Dd, q_eq, q_dot_eq, tau_eq),
    #     q_0=q_0,
    #     q_dot_0=q_dot_0,
    #     ctrl_fb=partial(lqr, Ad, Bd, Q, R),
    # )

    dt = CTRL_DT
    q_eq = np.array([0, 0, 0, np.pi/2])
    q_dot_eq = np.zeros(4)
    q_ddot_eq = np.zeros(4)

    t_uncontrolled, y_uncontrolled = simulate_uncontrolled_system(physical_parameters, t_ts_uncontrolled,
                                                                  sim_length_uncontrolled, q_0, q_dot_0)
    t_lqr_cont, y_lqr_cont = control_with_lqr_continuous(physical_parameters, t_ts, Q, R,
                                                         q_0, q_dot_0, q_eq, q_dot_eq, q_ddot_eq)
    t_lqr_discrete, y_lqr_discrete = control_with_lqr_discrete(physical_parameters, t_ts, dt, Q, R,
                                                               q_0, q_dot_0, q_eq, q_dot_eq, q_ddot_eq)

    t = t_lqr_discrete
    y = y_lqr_discrete

    plot_errors(t, y[:4, :], q_eq)
    plot_trajectories(t, y[:4, :], q_eq)
    link_ends_list = np.array([np.zeros_like(calculate_link_endpoints(physical_parameters, q_0))] * sim_length)
    for i in range(sim_length):
        # q_i = sim_ts['q_ts'][i]
        q_i = y[:4, i]
        link_ends_list[i] = calculate_link_endpoints(physical_parameters, q_i)
    animation.animate_mechanism_3d(link_ends_list, dt)

    pass


