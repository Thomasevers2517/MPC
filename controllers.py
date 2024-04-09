import numpy as np
from scipy import linalg
import control as ct

from functools import partial
from typing import Callable, Dict, Optional, Tuple
from numpy.typing import NDArray

from physical_modelling.dynamics import continuous_inverse_dynamics, continuous_state_space_dynamics
from physical_modelling.linearisation import continuous_linear_state_space_representation, cont2discrete_zoh


def lqr(A, B, Q, R, q, q_dot, q_eq, q_dot_eq):
    x = np.concatenate([q, q_dot])
    x_eq = np.concatenate([q_eq, q_dot_eq])

    K, S, E = ct.dlqr(A, B, Q, R)
    tau = - K @ (x - x_eq)

    return tau


def control_with_lqr_continuous(physical_parameters, t_ts, Q, R, q_0, q_dot_0, q_eq, q_dot_eq, q_ddot_eq):
    A, B, C, D = continuous_linear_state_space_representation(physical_parameters)

    x_0 = np.concatenate([q_0, q_dot_0])

    x_eq = np.concatenate([q_eq, q_dot_eq])
    # tau_eq = continuous_inverse_dynamics(physical_parameters, q_eq, q_dot_eq, q_ddot_eq)[[0, 3]]
    # sys_cont_open_loop = ct.linearize(sys, x_eq.tolist(), tau_eq.tolist())
    sys_cont_open_loop = ct.ss(A, B, C, D)

    K_cont, _, _ = ct.lqr(sys_cont_open_loop, Q, R)
    controller_cont, sys_cont_closed_loop = ct.create_statefbk_iosystem(sys_cont_open_loop, K_cont)

    x_des = x_eq
    x_des_and_tau_des = x_des.tolist() + [0, 0]
    t, y = ct.input_output_response(sys_cont_closed_loop, t_ts, x_des_and_tau_des, x_0)

    return t, y


def control_with_lqr_discrete(physical_parameters, t_ts, dt, Q, R, q_0, q_dot_0, q_eq, q_dot_eq, q_ddot_eq):
    A, B, C, D = continuous_linear_state_space_representation(physical_parameters)
    Ad, Bd, Cd, Dd = cont2discrete_zoh(dt, A, B, C, D)

    sys_cont = ct.ss(A, B, C, D)
    sys_discrete_open_loop = ct.ss(Ad, Bd, Cd, Dd, dt)
    # TODO: are they the same? haven't checked yet
    # sys_discrete_open_loop = ct.sample_system(sys_cont, dt)

    x_0 = np.concatenate([q_0, q_dot_0])
    x_eq = np.concatenate([q_eq, q_dot_eq])

    K_discrete, _, _ = ct.dlqr(sys_discrete_open_loop, Q, R)
    controller_discrete, sys_discrete_closed_loop = ct.create_statefbk_iosystem(sys_discrete_open_loop, K_discrete)

    x_des = x_eq
    x_des_and_tau_des = x_des.tolist() + [0, 0]
    t, y = ct.input_output_response(sys_discrete_closed_loop, t_ts, x_des_and_tau_des, x_0)

    return t, y
