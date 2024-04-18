import numpy as np

from typing import Dict, Tuple
from numpy.typing import NDArray
from functools import partial

# from .forward_kinematics import reduced_forward_kinematics, extended_reduced_forward_kinematics, forward_kinematics
from .dynamical_matrices import dynamical_matrices
from utils import rk4_step


def continuous_forward_dynamics(
    physical_parameters: dict,
    q: NDArray,
    q_dot: NDArray,
    tau: NDArray = np.zeros((2,)),
) -> NDArray:
    M, C, g = dynamical_matrices(physical_parameters, q, q_dot)

    tau_extended = np.insert(tau, 1, np.zeros(2)) if len(tau) == 2 else tau

    # link angular accelerations
    q_dd = np.linalg.inv(M) @ (tau_extended - C @ q_dot - g)

    return q_dd


def continuous_inverse_dynamics(
    physical_parameters: dict,
    q: NDArray,
    q_dot: NDArray,
    q_ddot: NDArray = np.zeros((2,)),
) -> NDArray:
    M, C, g = dynamical_matrices(physical_parameters, q, q_dot)

    # link torques
    tau = M @ q_ddot + C @ q_dot + g

    return tau


def continuous_state_space_dynamics(
    physical_parameters: dict,
    x: NDArray,
    tau: NDArray,
    available_states: NDArray = None
) -> Tuple[NDArray, NDArray]:
    if available_states is None:
        available_states = np.arange(len(x))
    if available_states.dtype != np.int32:
        available_states = np.array(available_states, dtype='int32')
    if len(tau) == 2:
        tau = np.insert(tau, 1, np.zeros(2))

    # split the state into link angles and velocities
    q, q_dot = np.split(x, 2)

    # compute the link angular accelerations
    q_ddot = continuous_forward_dynamics(physical_parameters, q, q_dot, tau)

    # time derivative of state
    dx_dt = np.concatenate([q_dot, q_ddot])

    # the system output are the two link angles
    y = x[available_states]

    return dx_dt, y


def discrete_forward_dynamics(
    physical_parameters: dict,
    dt: NDArray,
    q_curr: NDArray,
    q_dot_curr: NDArray,
    tau: NDArray = np.zeros((2,)),
) -> Tuple[NDArray, NDArray, NDArray]:
    q_ddot = continuous_forward_dynamics(physical_parameters, q_curr, q_dot_curr, tau)

    x_next = rk4_step(
        # generate `ode_fun`, which conforms to the signature ode_fun(x) -> dx_dt
        ode_fun=lambda _x: partial(continuous_state_space_dynamics, physical_parameters, tau=tau)(_x)[0],
        x=np.concatenate([q_curr, q_dot_curr]),  # concatenate the current state
        dt=dt,  # time step
    )
    q_next, q_dot_next = np.split(x_next, 2)

    # check validity of next step and if it would be impossible, force next joint angles to be the current ones
    L = physical_parameters['L']
    q_1 = q_next[0]
    q_2 = q_next[3]
    p_B = np.array([L * np.cos(q_1), L * np.sin(q_1)])
    p_D = np.array([2 * L + L * np.cos(q_2), L * np.sin(q_2)])
    if np.linalg.norm(p_B - p_D) > 2 * L:
        q_next[[0, 3]] = q_curr[[0, 3]]
        raise ValueError("Next step is invalid")

    return q_next, q_dot_next, q_ddot


# def continuous_linear_state_space_representation(
#     physical_parameters: dict,
#     q_eq: NDArray,
#     q_dot_eq: NDArray = np.zeros((2,)),
#     tau_eq: NDArray = np.zeros((2,)),
# ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
#     pass


def continuous_closed_loop_linear_state_space_representation(
    physical_parameters: dict,
    q_eq: NDArray,
    q_dot_eq: NDArray,
    tau_eq: NDArray,
    q_des: NDArray,
    q_dot_des: NDArray,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    pass
