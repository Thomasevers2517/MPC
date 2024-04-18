import numpy as np
from numpy.typing import NDArray
from typing import Dict, Callable
import control as ct
from functools import partial
from scipy.integrate import solve_ivp

import physical_modelling.initialisation as init
import animation
from physical_modelling.forward_kinematics import calculate_link_endpoints, reduced_forward_kinematics, generate_ellipse_trajectory
from physical_modelling.dynamics import continuous_inverse_dynamics, continuous_state_space_dynamics
from physical_modelling.simulation import simulate_uncontrolled_system
from physical_modelling.linearisation import continuous_linear_state_space_representation, cont2discrete_zoh
from controllers import control_with_lqr_continuous, control_with_lqr_discrete, ConstraintConstants, mpc, discrete_input_for_continuous_nonlinear_simulation, lqr_input
from plot_results import *
from time import time


# Control parameters
CTRL_DT = 0.01
# LQR
Q_LQR = np.diag([7, 6, 6, 7, 0.0001, 0.0001, 0.0001, 0.0001])
R_LQR = np.diag([18, 18])
# MPC
N = 10
Q = 10 * np.eye(8)
Q[0, 0] = 500
Q[3, 3] = 500
R = 10 * np.eye(2)
# Q = Q_LQR
# R = R_LQR
P = None            # set-point tracking
# P = 20 * np.eye(8)
# P = 20 * np.eye(8)  # trajectory tracking

# Constraints
# U_MAX = 4.47e-3
U_MAX = 0.1
MAX_LINEARIZATION_ERROR = np.deg2rad(5)
MAX_LINEARIZATION_ERROR_FINAL = np.deg2rad(5)
MAX_ERROR_Q_DOT_FINAL = 0.1

# Trajectory tracking
TRAJECTORY_TRACKING = False
OMEGA = 0.1
R_X = 0.1
R_Y = R_X
INCLINATION = 0

# Simulation parameters
SIM_DT = 0.01    # s
SIM_CONTINUOUS_DT = 1e-5
SIM_DURATION = 2 + SIM_DT  # s

# Initial conditions
Q_1_0 = 0.05                   # [rad]
PHI_0 = np.deg2rad(2)       # [rad]
THETA_0 = np.deg2rad(-2)    # [rad]
Q_2_0 = np.pi / 2 + 0.05          # [rad]
Q_1_DOT_0 = 0               # [rad]
PHI_DOT_0 = 0               # [rad]
THETA_DOT_0 = 0             # [rad]
Q_2_DOT_0 = 0               # [rad]


def odefun(t: float, x: NDArray, params: Dict, u_fun: Callable[[float, NDArray], NDArray]) -> NDArray:
    u = u_fun(t, x)
    dx_dt, _ = continuous_state_space_dynamics(params, x, u)
    return dx_dt


def discretesys(t, x, x_ref, params):
    A, B, C, D = continuous_linear_state_space_representation(params)
    Ad, Bd, _, _ = cont2discrete_zoh(0.001, A, B, C, D)
    x_e_next = Ad @ (x - x_ref)
    x_next = x_ref + x_e_next
    return x_next


if __name__ == "__main__":
    # ===================== Initialisation =====================
    # physical parameters
    physical_parameters = init.initialize_physical_parameters()

    # time instances
    t_ts, sim_length = init.generate_t_ts(SIM_DT, SIM_DURATION)
    t_ts_cont, sim_length_cont = init.generate_t_ts(SIM_CONTINUOUS_DT, SIM_DURATION)
    t_ts_uncontrolled, sim_length_uncontrolled = init.generate_t_ts(SIM_CONTINUOUS_DT, 0.65)  # singularity at t=0.65 s
    dt = CTRL_DT
    # exclude the last N time instances for the MPC simulation
    M = sim_length - N
    t_ts_MPC = t_ts[:M]

    # initial conditions
    q_0, q_dot_0 = init.set_initial_conditions(Q_1_0, PHI_0, THETA_0, Q_2_0,
                                               Q_1_DOT_0, PHI_DOT_0, THETA_DOT_0, Q_2_DOT_0)
    x_0 = np.concatenate([q_0, q_dot_0])
    y_0 = x_0

    # equilibria
    q_eq = np.array([0, 0, 0, np.pi/2])
    q_dot_eq = np.zeros(4)
    q_ddot_eq = np.zeros(4)
    tau_eq = continuous_inverse_dynamics(physical_parameters, q_eq, q_dot_eq, q_ddot_eq)

    x_eq = np.concatenate([q_eq, q_dot_eq])
    u_eq = tau_eq[[0, 3]]

    # reference trajectory
    if TRAJECTORY_TRACKING:
        _, _, x_center, y_center = reduced_forward_kinematics(physical_parameters, q_eq)
        y_ref, u_ref = generate_ellipse_trajectory(physical_parameters, t_ts_MPC, OMEGA, x_center, y_center, R_X)
        # y_ref, u_ref = generate_ellipse_trajectory(physical_parameters, t_ts, OMEGA,
        #                                            x_center, y_center, R_X, R_Y, INCLINATION)
    else:
        y_ref = x_eq
        u_ref = u_eq

    # constraints
    constraints = ConstraintConstants(u_lim=U_MAX * np.ones(2),
                                      max_linearization_error=MAX_LINEARIZATION_ERROR * np.ones(2),
                                      max_linearization_error_final=MAX_LINEARIZATION_ERROR_FINAL * np.ones(2),
                                      max_error_q_dot_final=MAX_ERROR_Q_DOT_FINAL * np.ones(4))

    # ===================== Simulation =====================
    # -------------------
    # uncontrolled system
    print("-" * 50)
    print("Simulating uncontrolled system...")
    t_0 = time()
    results_uncontrolled = solve_ivp(odefun, t_span=(0, 0.65), y0=x_0, t_eval=t_ts_uncontrolled,
                                     args=(physical_parameters, lambda t_, x_: np.zeros(2)))
    t_ts_, _ = init.generate_t_ts(SIM_DT, 0.65)
    results_uncontrolled_discrete = solve_ivp(discretesys, t_span=(0, 0.65), y0=x_0, t_eval=t_ts_, args=(x_eq, physical_parameters))
    print(f"Uncontrolled system simulation finished in {time() - t_0:.2f} s")

    # -------------------
    # control with LQR
    # continuous system
    print("-" * 50)
    print("Simulating LQR controlled continuous system...")
    t_0 = time()
    u_LQR_cont_fun = partial(lqr_input, x_eq=y_ref, u_eq=u_ref, Q=Q_LQR, R=R_LQR,
                             params=physical_parameters, continuous=True)
    dynamics_LQR_cont = partial(odefun, params=physical_parameters, u_fun=u_LQR_cont_fun)
    results_lqr_continuous = solve_ivp(dynamics_LQR_cont, t_span=(0, SIM_DURATION), y0=x_0, t_eval=t_ts_cont)
    print(f"LQR controlled continuous system simulation finished in {time() - t_0:.2f} s")

    # discrete system
    print("-" * 50)
    print("Simulating LQR controlled discrete system...")
    u_LQR_discrete_fun = partial(lqr_input, x_eq=y_ref, u_eq=u_ref, Q=Q_LQR, R=R_LQR,
                                 params=physical_parameters, dt=SIM_DT)
    dynamics_LQR_discrete = partial(odefun, params=physical_parameters, u_fun=u_LQR_discrete_fun)
    results_lqr_discrete = solve_ivp(dynamics_LQR_cont, t_span=(0, SIM_DURATION), y0=x_0, t_eval=t_ts)
    print(f"LQR controlled discrete system simulation finished in {time() - t_0:.2f} s")

    # -------------------
    # control with MPC
    # discrete system
    print("-" * 50)
    print("Simulating MPC controlled discrete system...")
    t_0 = time()
    t_MPC, y_MPC, u_MPC = mpc(t_ts, y_ref, u_ref, x_0, N, dt, Q, R, P, constraints, physical_parameters)
    print(f"MPC controlled discrete system simulation finished in {time() - t_0:.2f} s")

    # DOES NOT WORK
    # # continuous system
    # print("-" * 50)
    # print("Simulating MPC controlled continuous system...")
    # t_0 = time()
    # u_MPC_cont_fun = partial(discrete_input_for_continuous_nonlinear_simulation, y_ref=y_ref, u_ref=u_ref, N=N, M=M,
    #                          dt=dt, Q=Q, R=R, P=P, constraint_constants=constraints, params=physical_parameters)
    # dynamics_MPC_cont = partial(odefun, params=physical_parameters, u_fun=u_MPC_cont_fun)
    # results_MPC_cont = solve_ivp(dynamics_MPC_cont, t_span=(0, SIM_DURATION - SIM_CONTINUOUS_DT * N),
    #                              y0=y_0, t_eval=t_ts_cont[:sim_length_cont - N])
    # print(f"MPC controlled continuous system simulation finished in {time() - t_0:.2f} s")
    # print("-" * 50)

    # ===================== Plotting =====================
    # t = t_lqr_discrete
    # y = y_lqr_discrete
    # t = t_uncontrolled
    # y = y_uncontrolled
    # t = results_uncontrolled_discrete.t
    # y = results_uncontrolled_discrete.y
    # sim_length = 65
    # t = results_lqr_discrete.t
    # y = results_lqr_discrete.y
    t = t_MPC
    y = y_MPC

    # plot_errors(t, y[:4, :], q_eq)
    # plot_error_mpc_vs_lqr(t_MPC, y_MPC, results_lqr_discrete.y[:, :M + 1], y_ref)

    Q_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    R_list = [0.001, 0.01, 0.1, 1, 10]
    R_list_for_u = [0.01, 1, 100]
    P_list = [1, 10, 100, 1000, 10000]
    dt_list = [0.001, 0.01]
    N_list = [3, 5, 10, 15]

    # plot_effect_of_mpc_weighting_matrices(t_ts, y_ref, u_ref, x_0, dt, N, Q, R, P, constraints, physical_parameters,
    #                                       "Q", Q_list, "figs/Q_effect.png")
    # plot_effect_of_mpc_weighting_matrices(t_ts, y_ref, u_ref, x_0, dt, N, Q, R, P, constraints, physical_parameters,
    #                                         "R", R_list, "figs/R_effect.png")
    # plot_effect_of_mpc_weighting_matrices(t_ts, y_ref, u_ref, x_0, dt, N, Q, R, P, constraints, physical_parameters,
    #                                       "P", P_list, "figs/P_effect.png")
    #
    # plot_effect_of_R_on_u(t_ts, y_ref, u_ref, x_0, dt, N, Q, P, constraints, physical_parameters,
    #                       R_list_for_u, "figs/R_effect_on_u.png")
    #
    # plot_effect_of_dt(t_ts, y_ref, u_ref, x_0, dt_list, N, Q, R, P, constraints,
    #                   physical_parameters, "figs/dt_effect.png")

    # plot_effect_of_mpc_horizon(t_ts, y_ref, u_ref, x_0, dt, Q, R, P, constraints, physical_parameters,
    #                            N_list, "figs/N_effect.png")

    plot_error_mpc_vs_lqr(t_MPC, y_MPC, results_lqr_discrete.y[:, :M + 1], y_ref, "figs/error_mpc_vs_lqr.png")

    # plot_trajectories(t, y[:4, :], q_eq)
    # link_ends_list = np.array([np.zeros_like(calculate_link_endpoints(physical_parameters, q_0))] * sim_length)
    # for i in range(sim_length):
    #     # q_i = sim_ts['q_ts'][i]
    #     q_i = y[:4, i]
    #     link_ends_list[i] = calculate_link_endpoints(physical_parameters, q_i)
    # animation.animate_mechanism_3d(link_ends_list, dt)

    pass


