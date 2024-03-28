import numpy as np

from typing import Tuple, Dict, Any
from numpy.typing import NDArray

from .forward_kinematics import reduced_forward_kinematics


def jacobian(physical_parameters: Dict, q: NDArray
) -> NDArray:
    L = physical_parameters['L']

    q_1 = q[0]
    q_2 = q[3]

    beta_1, beta_2, _, _ = reduced_forward_kinematics(physical_parameters, q)

    den = np.sin(q_2 - q_1 + beta_2 - beta_1)

    J_11 = np.sin(q_2 + beta_2) * np.sin(beta_1) / den
    J_14 = - np.sin(q_1 + beta_1) * np.sin(beta_2) / den
    J_21 = - np.cos(q_2 + beta_2) * np.sin(beta_1) / den
    J_24 = - np.cos(q_1 + beta_1) * np.sin(beta_2) / den

    J = L * np.array([[J_11, 0, 0, J_14],
                      [J_21, 0, 0, J_24],
                      [0,    1, 0, 0],
                      [0,    0, 1, 0]])

    return J


def jacobian_beta(physical_parameters: Dict, q: NDArray
) -> NDArray:
    q_1 = q[0]
    q_2 = q[3]

    beta_1, beta_2, _, _ = reduced_forward_kinematics(physical_parameters, q)

    den = np.sin(q_1 - q_2 + beta_1 - beta_2)

    J_beta_11 = - np.sin(q_1 - q_2 - beta_2) / den - 1
    J_beta_12 = - np.sin(beta_2) / den
    J_beta_21 = np.sin(beta_2) / den
    J_beta_22 = - np.sin(q_1 - q_2 + beta_1) / den - 1

    J_beta = np.array([[J_beta_11, J_beta_12],
                       [J_beta_21, J_beta_22]])

    return J_beta


def jacobian_dot(physical_parameters: Dict, q: NDArray, q_dot: NDArray
) -> NDArray:
    from physical_modelling.forward_kinematics import extended_reduced_forward_kinematics

    L = physical_parameters['L']

    q_1 = q[0]
    q_2 = q[3]
    beta_1, beta_2, _, _ = reduced_forward_kinematics(physical_parameters, q)

    q_1_dot = q_dot[0]
    q_2_dot = q_dot[3]
    beta_1_dot, beta_2_dot, _, _ = extended_reduced_forward_kinematics(physical_parameters, q, q_dot)

    num_1 = q_1 + beta_1
    num_2 = q_2 + beta_2
    num_3 = beta_1
    num_4 = beta_2
    den = q_2 - q_1 + beta_2 - beta_1

    num_1_dot = q_2_dot + beta_2_dot
    num_2_dot = q_1_dot + beta_1_dot
    num_3_dot = beta_2_dot
    num_4_dot = beta_1_dot
    den_dot = q_2_dot - q_1_dot + beta_2_dot - beta_1_dot

    arg_den = den
    arg_den_dot = den_dot

    # 1st column
    arg_num_1 = num_2
    arg_num_2 = num_3
    arg_num_1_dot = num_2_dot
    arg_num_2_dot = num_3_dot
    J_dot_11 = ((np.cos(arg_num_1) * arg_num_1_dot * np.sin(arg_num_2) + np.sin(arg_num_1) * np.cos(
        arg_num_2) * arg_num_2_dot) * np.sin(arg_den) - np.sin(arg_num_1) * np.sin(arg_num_2) * np.cos(
        arg_den) * arg_den_dot) / np.sin(arg_den) ** 2
    J_dot_21 = ((np.sin(arg_num_1) * arg_num_1_dot * np.sin(arg_num_2) - np.cos(arg_num_1) * np.cos(
        arg_num_2) * arg_num_2_dot) * np.sin(arg_den) + np.cos(arg_num_1) * np.sin(arg_num_2) * np.cos(
        arg_den) * arg_den_dot) / np.sin(arg_den) ** 2

    # 4th column
    arg_num_1 = num_1
    arg_num_2 = num_4
    arg_num_1_dot = num_1_dot
    arg_num_2_dot = num_4_dot
    J_dot_14 = - ((np.cos(arg_num_1) * arg_num_1_dot * np.sin(arg_num_2) + np.sin(arg_num_1) * np.cos(
        arg_num_2) * arg_num_2_dot) * np.sin(arg_den) - np.sin(arg_num_1) * np.sin(arg_num_2) * np.cos(
        arg_den) * arg_den_dot) / np.sin(arg_den) ** 2
    J_dot_24 = - ((np.sin(arg_num_1) * arg_num_1_dot * np.sin(arg_num_2) - np.cos(arg_num_1) * np.cos(
        arg_num_2) * arg_num_2_dot) * np.sin(arg_den) + np.cos(arg_num_1) * np.sin(arg_num_2) * np.cos(
        arg_den) * arg_den_dot) / np.sin(arg_den) ** 2

    J_dot = L * np.array([[J_dot_11, 0, 0, J_dot_14],
                          [J_dot_21, 0, 0, J_dot_24],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0]])

    return J_dot


def _jacobian_beta_dot_ii(arg_num, arg_den, arg_num_dot, arg_den_dot):
    J_dot_ii = (-np.cos(arg_num) * arg_num_dot * np.cos(arg_den)
                + np.sin(arg_num) * np.cos(arg_den) * arg_den_dot) / np.sin(arg_den) ** 2
    return J_dot_ii


def jacobian_beta_dot(physical_parameters: Dict, q: NDArray, q_dot: NDArray
) -> NDArray:
    from physical_modelling.forward_kinematics import extended_reduced_forward_kinematics

    q_1 = q[0]
    q_2 = q[3]
    beta_1, beta_2, _, _ = reduced_forward_kinematics(physical_parameters, q)

    q_1_dot = q_dot[0]
    q_2_dot = q_dot[3]
    beta_1_dot, beta_2_dot, _, _ = extended_reduced_forward_kinematics(physical_parameters, q, q_dot)

    num11 = q_1 - q_2 - beta_2
    num12 = beta_2
    num21 = beta_1
    num22 = q_1 - q_2 + beta_1
    den = q_1 - q_2 + beta_1 - beta_2
    num11_dot = q_1_dot - q_2_dot - beta_2_dot
    num12_dot = beta_2_dot
    num21_dot = beta_1_dot
    num22_dot = q_1_dot - q_2_dot + beta_1_dot
    den_dot = q_1_dot - q_2_dot + beta_1_dot - beta_2_dot

    J_beta_11_dot = _jacobian_beta_dot_ii(num11, den, num11_dot, den_dot)
    J_beta_12_dot = _jacobian_beta_dot_ii(num12, den, num12_dot, den_dot)
    J_beta_21_dot = _jacobian_beta_dot_ii(num21, den, num21_dot, den_dot)
    J_beta_22_dot = _jacobian_beta_dot_ii(num22, den, num22_dot, den_dot)

    J_beta_dot = np.array([[J_beta_11_dot, J_beta_12_dot],
                           [J_beta_21_dot, J_beta_22_dot]])

    return J_beta_dot
