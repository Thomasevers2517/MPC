import numpy as np
from scipy import linalg

from functools import partial
from typing import Callable, Dict, Optional, Tuple
from numpy.typing import NDArray

from .dynamics import continuous_forward_dynamics
# from controllers import lqr


def continuous_linear_state_space_representation(
        physical_parameters: Dict,
        # continuous_forward_dynamics_fn: Callable,
        # q_eq: NDArray,
        # q_dot_eq: NDArray = np.zeros((2,)),
        # tau_eq: NDArray = np.zeros((2,)),
        # *args_dynamics,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    I_1 = physical_parameters['I_1']
    I_2 = physical_parameters['I_2']
    I_3 = physical_parameters['I_3']
    I_4 = physical_parameters['I_4']
    I_px = physical_parameters['I_px']
    I_pz = physical_parameters['I_pz']
    m_1 = physical_parameters['m_1']
    m_2 = physical_parameters['m_2']
    m_3 = physical_parameters['m_3']
    m_4 = physical_parameters['m_4']
    m_p = physical_parameters['m_p']
    l_c1 = physical_parameters['l_c1']
    l_c2 = physical_parameters['l_c2']
    l_c3 = physical_parameters['l_c3']
    l_c4 = physical_parameters['l_c4']
    l_cp = physical_parameters['l_cp']
    L = physical_parameters['L']
    g_0 = physical_parameters['g_0']

    h_1 = I_2 + I_4 + I_pz + L ** 2 * m_3 + L ** 2 * m_p + l_c2 ** 2 * m_2 + l_c4 ** 2 * m_4
    h_2 = I_1 + I_3 + L ** 2 * m_2 + L ** 2 * m_p + l_c1 ** 2 * m_1 + l_c3 ** 2 * m_3
    h_3 = (I_2 * I_px + I_4 * I_px + I_px * I_pz + I_px * L ** 2 * m_3 + I_px * L ** 2 * m_p
           + I_2 * l_cp ** 2 * m_p + I_px * l_c2 ** 2 * m_2 + I_4 * l_cp ** 2 * m_p + I_px * l_c4 ** 2 * m_4
           + I_pz * l_cp ** 2 * m_p + L ** 2 * l_cp ** 2 * m_3 * m_p + l_c2 ** 2 * l_cp ** 2 * m_2 * m_p
           + l_c4 ** 2 * l_cp ** 2 * m_4 * m_p)
    h_4 = (I_1 * I_px + I_3 * I_px + I_px * L ** 2 * m_2 + I_px * L ** 2 * m_p + I_1 * l_cp ** 2 * m_p
           + I_px * l_c1 ** 2 * m_1 + I_3 * l_cp ** 2 * m_p + I_px * l_c3 ** 2 * m_3
           + L ** 2 * l_cp ** 2 * m_2 * m_p + l_c1 ** 2 * l_cp ** 2 * m_1 * m_p + l_c3 ** 2 * l_cp ** 2 * m_3 * m_p)

    a_62 = g_0 * l_cp * m_p * h_1 / h_3
    a_82 = g_0 * L * l_cp ** 2 * m_p ** 2 / h_3
    a_53 = -g_0 * L * l_cp ** 2 * m_p ** 2 / h_4
    a_73 = g_0 * l_cp * m_p * h_2 / h_4

    b_51 = (m_p * l_cp ** 2 + I_px) / h_4
    b_71 = -L * l_cp * m_p / h_4
    b_62 = L * l_cp * m_p / h_3
    b_82 = (m_p * l_cp ** 2 + I_px) / h_3

    A = np.array([[0, 0,    0,    0, 1, 0, 0, 0],
                  [0, 0,    0,    0, 0, 1, 0, 0],
                  [0, 0,    0,    0, 0, 0, 1, 0],
                  [0, 0,    0,    0, 0, 0, 0, 1],
                  [0, 0,    a_53, 0, 0, 0, 0, 0],
                  [0, a_62, 0,    0, 0, 0, 0, 0],
                  [0, 0,    a_73, 0, 0, 0, 0, 0],
                  [0, a_82, 0,    0, 0, 0, 0, 0],
                  ])
    B = np.array([[0,    0],
                  [0,    0],
                  [0,    0],
                  [0,    0],
                  [b_51, 0],
                  [0,    b_62],
                  [b_71, 0],
                  [0,    b_82]])
    C = np.eye(8)
    D = np.zeros((8, 2))

    return A, B, C, D


def cont2discrete_zoh(
    dt: float,
    A: NDArray,
    B: NDArray,
    C: NDArray,
    D: NDArray,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    em_upper = np.hstack((A, B))

    # Need to stack zeros under the A and B matrices
    em_lower = np.hstack((np.zeros((B.shape[1], A.shape[0])),
                           np.zeros((B.shape[1], B.shape[1]))))

    em = np.vstack((em_upper, em_lower))
    ms = linalg.expm(dt * em)

    # Dispose of the lower rows
    ms = ms[:A.shape[0], :]

    Ad = ms[:, 0:A.shape[1]]
    Bd = ms[:, A.shape[1]:]

    Cd = C
    Dd = D

    return Ad, Bd, Cd, Dd


def linearized_discrete_forward_dynamics(
    Ad: NDArray,
    Bd: NDArray,
    Cd: NDArray,
    Dd: NDArray,
    q_eq: NDArray,
    q_dot_eq: NDArray,
    tau_eq: NDArray,
    dt: float,
    q: NDArray,
    q_dot: NDArray,
    tau: NDArray,
) -> Tuple[NDArray, NDArray, NDArray]:
    x = np.concatenate([q, q_dot])
    x_eq = np.concatenate([q_eq, q_dot_eq])
    x_next = Ad @ (x - x_eq) + Bd @ (tau - tau_eq)

    q_next = x_next[0:4]
    q_dot_next = x_next[4:]
    q_ddot = (q_dot_next - q_dot) / dt

    return q_next, q_dot_next, q_ddot


def closed_loop_fb_continuous_forward_dynamics(
        physical_parameters: Dict,
        q: NDArray,
        q_dot: NDArray,
        tau_ext: NDArray = np.zeros(2),
        q_des: NDArray = np.array([0, 0, 0, np.pi/2]),
        q_dot_des: NDArray = np.zeros(4),
) -> NDArray:
    # tau_fb = ctrl_fb_pd(q, q_dot, q_des, q_dot_des,)
    # tau_fb = lqr(q, q_dot, q_des, q_dot_des,)
    tau_fb = np.zeros_like(tau_ext)
    tau = tau_fb + tau_ext

    q_ddot = continuous_forward_dynamics(physical_parameters, q, q_dot, tau)

    return q_ddot
    
