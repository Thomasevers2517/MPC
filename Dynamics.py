import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from typing import Dict, Tuple
from numpy.typing import ArrayLike


def S(angle):
    return np.sin(angle)


def C(angle):
    return np.cos(angle)


def direct_kinematics(physical_parameters: Dict, q: ArrayLike
) -> Tuple[float, float, float, float]:
    L = physical_parameters['lengths'][5]

    q_1 = q[0]
    q_2 = q[3]

    fraction_aux = C(q_1 - q_2) + 2 * C(q_1) - 2 * C(q_2)
    if np.isnan(fraction_aux):
        pass
    sqrt_term = np.sqrt(-(fraction_aux - 1) / (fraction_aux - 3))

    x_b1 = L / 2 * (2 + C(q_1) + C(q_2)) + L / 2 * (S(q_1) - S(q_2)) * sqrt_term
    x_b2 = L / 2 * (2 + C(q_1) + C(q_2)) - L / 2 * (S(q_1) - S(q_2)) * sqrt_term

    y_b1 = L / 2 * (S(q_1) + S(q_2)) + L / 2 * (C(q_2) - C(q_1) + 2) * sqrt_term
    y_b2 = L / 2 * (S(q_1) + S(q_2)) - L / 2 * (C(q_2) - C(q_1) + 2) * sqrt_term

    # select correct x_b, y_v TODO
    x_b = x_b1
    y_b = y_b1

    beta_1 = np.arctan2(y_b - L * S(q_1), x_b - L * C(q_1)) - q_1
    beta_2 = np.arctan2(y_b - L * S(q_2), x_b - (2 * L + L * C(q_2))) - q_2

    return beta_1, beta_2, x_b, y_b


def calculate_J_beta(physical_parameters: Dict, q: ArrayLike
) -> ArrayLike:
    q_1 = q[0]
    q_2 = q[3]

    beta_1, beta_2, _, _ = direct_kinematics(physical_parameters, q)

    den = S(q_1 - q_2 + beta_1 - beta_2)

    J_beta_11 = - S(q_1 - q_2 - beta_2) / den - 1
    J_beta_12 = - S(beta_2) / den
    J_beta_21 = S(beta_2) / den
    J_beta_22 = - S(q_1 - q_2 + beta_1) / den - 1

    J_beta = np.array([[J_beta_11, J_beta_12],
                       [J_beta_21, J_beta_22]])

    return J_beta


def _calculate_J_dot_ii(arg_num, arg_den, arg_num_dot, arg_den_dot):
    J_dot_ii = (-C(arg_num) * arg_num_dot * C(arg_den) + S(arg_num) * C(arg_den) * arg_den_dot) / S(arg_den) ** 2
    return J_dot_ii


def calculate_J_beta_dot(physical_parameters: Dict, q: ArrayLike, q_dot: ArrayLike
) -> ArrayLike:
    q_1 = q[0]
    q_2 = q[3]
    beta_1, beta_2, _, _ = direct_kinematics(physical_parameters, q)

    q_1_dot = q_dot[0]
    q_2_dot = q_dot[3]
    beta_1_dot, beta_2_dot = differential_kinematics(physical_parameters, q, q_dot)

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

    J_beta_11_dot = _calculate_J_dot_ii(num11, den, num11_dot, den_dot)
    J_beta_12_dot = _calculate_J_dot_ii(num12, den, num12_dot, den_dot)
    J_beta_21_dot = _calculate_J_dot_ii(num21, den, num21_dot, den_dot)
    J_beta_22_dot = _calculate_J_dot_ii(num22, den, num22_dot, den_dot)

    J_beta_dot = np.array([[J_beta_11_dot, J_beta_12_dot],
                           [J_beta_21_dot, J_beta_22_dot]])

    return J_beta_dot


def differential_kinematics(physical_parameters: Dict, q: ArrayLike, q_dot: ArrayLike
) -> Tuple[float, float]:
    q_dot_red = q_dot[[0, 3]]

    J_beta = calculate_J_beta(physical_parameters, q)

    beta_dot = J_beta @ q_dot_red

    beta_1_dot = beta_dot[0]
    beta_2_dot = beta_dot[1]

    return beta_1_dot, beta_2_dot


def calculate_M(physical_parameters: dict, q: ArrayLike
) -> ArrayLike:
    inertias = physical_parameters['inertias']
    masses = physical_parameters['masses']
    lengths = physical_parameters['lengths']

    I_1 = inertias[0]
    I_2 = inertias[1]
    I_3 = inertias[2]
    I_4 = inertias[3]
    I_px = inertias[4]
    I_pz = inertias[5]

    m_1 = masses[0]
    m_2 = masses[1]
    m_3 = masses[2]
    m_4 = masses[3]
    m_p = masses[4]

    l_c1 = lengths[0]
    l_c2 = lengths[1]
    l_c3 = lengths[2]
    l_c4 = lengths[3]
    l_cp = lengths[4]
    L = lengths[5]

    phi = q[1]
    theta = q[2]

    beta_1, beta_2, _, _ = direct_kinematics(physical_parameters, q)

    m_11 = (I_1 + I_2 + I_px + L ** 2 * m_2 + 2 * L ** 2 * m_p + l_c1 ** 2 * m_1 + l_c2 ** 2 * m_2 + l_cp ** 2 * m_p
            - I_px * C(phi) ** 2 * C(theta) ** 2
            + I_pz * C(phi) ** 2 * C(theta) ** 2
            + 2 * L ** 2 * m_p * C(beta_1)
            - l_cp ** 2 * m_p * C(phi) ** 2 * C(
                theta) ** 2  # no m_p in the paper but there has to be a mass term for correct dimension
            + 2 * L * l_c2 * m_2 * C(beta_1)
            + 4 * L * l_cp * m_p * C(beta_1 / 2) ** 2 * S(theta)
            + 2 * L * l_cp * m_p * S(beta_1) * C(theta) * S(phi))
    m_12 = (I_2 + I_px + L ** 2 * m_p + l_c2 ** 2 * m_2 + l_cp ** 2 * m_p
            + L ** 2 * m_p * C(beta_1) +
            I_pz * C(phi) ** 2 * C(theta) ** 2
            - I_px * C(phi) ** 2 * C(theta) ** 2
            - l_cp ** 2 * m_p * C(phi) ** 2 * C(theta) ** 2
            + L * l_c2 * m_2 * C(beta_1)
            + 2 * L * l_cp * m_p * S(theta)
            + L * l_cp * m_p * C(beta_1) * S(theta)
            + L * l_cp * m_p * S(beta_1) * C(theta) * S(phi))
    m_13 = (-C(phi) * C(theta) * L * l_cp * m_p * (1 + C(beta_1))
            - C(phi) * C(theta) * (I_px - I_pz + l_cp ** 2 * m_p) * S(theta))
    m_14 = (l_cp ** 2 * m_p * S(phi)
            + 2 * L * l_cp * m_p * S(phi) * S(theta) * C(beta_1 / 2) ** 2
            + I_px * S(phi) + L * l_cp * m_p * S(beta_1) * C(theta))
    m_22 = (I_2 + I_px + L ** 2 * m_p + l_c2 ** 2 * m_2 + l_cp ** 2 * m_p
            - (I_px - I_pz + l_cp ** 2 * m_p) * C(phi) ** 2 * C(theta) ** 2
            + 2 * L * l_cp * m_p * S(theta))
    m_23 = (-C(phi) * C(theta) * (S(theta) * (I_px - I_pz + l_cp ** 2 * m_p) + L * l_cp * m_p))
    m_24 = (S(phi) * (I_px + l_cp ** 2 * m_p + L * l_cp * m_p * S(theta)))
    m_33 = (I_pz + (I_px - I_pz + l_cp ** 2 * m_p) * C(theta) ** 2)
    m_44 = (I_pz + l_cp ** 2 * m_p)
    m_55 = (m_4 * l_c4 ** 2 + m_3 * (L ** 2 + l_c3 ** 2 + 2 * L * l_c3 * C(beta_2)) + I_4 + I_3)
    m_56 = (m_3 * (l_c3 ** 2 + L * l_c3 * C(beta_2)) + I_3)
    m_66 = (m_3 * l_c3 ** 2 + I_3)

    # not specified in paper (calculate/ask!) TODO
    m_34 = 0

    M = np.array([[m_11, m_12, m_13, m_14, 0,    0],
                  [m_12, m_22, m_23, m_24, 0,    0],
                  [m_13, m_23, m_33, m_34, 0,    0],
                  [m_14, m_24, m_34, m_44, 0,    0],
                  [0,    0,    0,    0,    m_55, m_56],
                  [0,    0,    0,    0,    m_56, m_66]])
    return M


def calculate_C(physical_parameters: dict, q: ArrayLike, q_dot: ArrayLike
) -> ArrayLike:
    inertias = physical_parameters['inertias']
    masses = physical_parameters['masses']
    lengths = physical_parameters['lengths']

    I_px = inertias[4]
    I_pz = inertias[5]

    m_1 = masses[0]
    m_2 = masses[1]
    m_3 = masses[2]
    m_4 = masses[3]
    m_p = masses[4]

    l_c1 = lengths[0]
    l_c2 = lengths[1]
    l_c3 = lengths[2]
    l_c4 = lengths[3]
    l_cp = lengths[4]
    L = lengths[5]

    q_1 = q[0]
    phi = q[1]
    theta = q[2]
    q_2 = q[3]

    q_1_dot = q_dot[0]
    phi_dot = q_dot[1]
    theta_dot = q_dot[2]
    q_2_dot = q_dot[3]

    beta_1, beta_2, _, _ = direct_kinematics(physical_parameters, q)
    beta_1_dot, beta_2_dot = differential_kinematics(physical_parameters, q, q_dot)

    a_1 = (L * l_cp * m_p * C(beta_1) * C(theta) * S(phi)
           - L * S(beta_1) * (l_c2 * m_2 + L * m_p + l_cp * m_p * S(theta)))
    a_2 = (C(phi) * C(theta) * L * l_cp * m_p * S(beta_1)
           + C(phi) * C(theta) * (I_px - I_pz + l_cp ** 2 * m_p) * C(theta) * S(phi))
    a_3 = (-L * l_cp * m_p * S(beta_1) * S(theta) * S(phi) + C(theta) * L * l_cp * m_p * (1 + C(beta_1))
           + C(theta) * (I_px - I_pz + l_cp ** 2 * m_p) * C(phi) ** 2 * S(theta))
    a_4 = (C(theta) * S(phi) * L * l_cp * m_p * (1 + C(beta_1))
           + C(theta) * S(phi) * (I_px - I_pz + l_cp ** 2 * m_p) * S(theta))
    a_5 = (1/2 * C(phi) * (I_px + l_cp ** 2 * m_p - (I_px - I_pz + l_cp ** 2 * m_p) * C(2 * theta))
           + 2 * C(phi) * L * l_cp * m_p * C(beta_1 / 2) ** 2 * S(theta))
    a_6 = L * l_cp * m_p * (2 * C(beta_1 / 2) ** 2 * C(theta) * S(phi) - S(beta_1) * S(theta))
    a_7 = (I_px - I_pz + l_cp ** 2 * m_p) * C(phi) * S(phi) * C(theta) ** 2
    a_8 = C(theta) * (L * l_cp * m_p + (I_px - I_pz + l_cp ** 2 * m_p) * C(phi) ** 2 * S(theta))
    a_9 = C(theta) * S(phi) * (L * l_cp * m_p + (I_px - I_pz + l_cp ** 2 * m_p) * S(theta))
    a_10 = (1/2 * C(phi) * (I_px + l_cp ** 2 * m_p - (I_px - I_pz + l_cp ** 2 * m_p) * C(2 * theta))
            + C(phi) * L * l_cp * m_p * C(beta_1 / 2) ** 2 * S(theta))
    a_11 = L * l_cp * m_p * C(theta) * S(phi)
    a_12 = -1/2 * C(phi) * (I_px + l_cp ** 2 * m_p + (I_px - I_pz + l_cp ** 2 * m_p) * C(2 * theta))
    a_13 = -(I_px - I_pz + l_cp ** 2 * m_p) * C(theta) * S(theta)

    c_11 = a_1 * beta_1_dot + a_2 * phi_dot + a_3 * theta_dot
    c_12 = a_1 * (q_1_dot + beta_1_dot) + a_2 * phi_dot + a_3 * theta_dot
    c_13 = a_2 * (q_1_dot + beta_1_dot) + a_4 * phi_dot + a_5 * theta_dot
    c_14 = a_3 * (q_1_dot + beta_1_dot) + a_5 * phi_dot + a_6 * theta_dot
    c_21 = -a_1 * q_1_dot + a_7 * phi_dot + a_8 * theta_dot
    c_22 = a_7 * phi_dot + a_8 * theta_dot
    c_23 = a_7 * (q_1_dot + beta_1_dot) + a_9 * phi_dot + a_10 * theta_dot
    c_24 = a_8 * (q_1_dot + beta_1_dot) + a_10 * phi_dot + a_11 * theta_dot
    c_31 = -a_2 * q_1_dot - a_7 * beta_1_dot + a_12 * theta_dot
    c_32 = -a_7 * (q_1_dot + beta_1_dot) + a_12 * theta_dot
    c_33 = a_13 * theta_dot
    c_34 = a_12 * (q_1_dot + beta_1_dot) + a_13 * phi_dot
    c_41 = -a_3 * q_1_dot - a_8 * beta_1_dot - a_12 * phi_dot
    c_42 = -a_8 * (q_1_dot + beta_1_dot) - a_12 * phi_dot
    c_43 = -c_34
    c_44 = 0
    c_55 = -m_3 * L * l_c3 * S(beta_2_dot) * beta_2_dot
    c_56 = -m_3 * L * l_c3 * S(beta_2_dot) * (q_2_dot + beta_2_dot)
    c_65 = m_3 * L * l_c3 * S(beta_2_dot) * q_2_dot
    c_66 = 0

    C_ = np.array([[c_11, c_12, c_13, c_14, 0,    0],
                   [c_21, c_22, c_23, c_24, 0,    0],
                   [c_31, c_32, c_33, c_34, 0,    0],
                   [c_41, c_42, c_43, c_44, 0,    0],
                   [0,    0,    0,    0,    c_55, c_56],
                   [0,    0,    0,    0,    c_65, c_66]])
    return C_


def calculate_g(physical_parameters: dict, q: ArrayLike
) -> ArrayLike:
    inertias = physical_parameters['inertias']
    masses = physical_parameters['masses']
    lengths = physical_parameters['lengths']

    m_p = masses[4]
    l_cp = lengths[4]

    phi = q[1]
    theta = q[2]

    _g = 9.81

    g = - m_p * _g * l_cp * np.array([0, S(phi) * C(theta), C(phi) * S(theta), 0])

    return g


def calculate_R(physical_parameters: dict, q: ArrayLike
) -> ArrayLike:
    J_beta = calculate_J_beta(physical_parameters, q)

    J_beta_11 = J_beta[0, 0]
    J_beta_12 = J_beta[0, 1]
    J_beta_21 = J_beta[1, 0]
    J_beta_22 = J_beta[1, 1]

    # R = np.insert(np.eye(4), [1, 4], np.insert(J_beta, [1, 1], np.zeros((2,2)), axis=1), axis=0)
    R = np.array([[1, 0, 0, 0],
                  [J_beta_11, 0, 0, J_beta_12],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [J_beta_21, 0, 0, J_beta_22]])

    return R


def calculate_R_dot(physical_parameters: dict, q: ArrayLike, q_dot: ArrayLike
) -> ArrayLike:
    J_beta_dot = calculate_J_beta_dot(physical_parameters, q, q_dot)

    J_beta_11 = J_beta_dot[0, 0]
    J_beta_12 = J_beta_dot[0, 1]
    J_beta_21 = J_beta_dot[1, 0]
    J_beta_22 = J_beta_dot[1, 1]

    # R_dot = np.insert(np.eye(4), [1, 4], np.insert(J_beta_dot, [1, 1], np.zeros((2,2)), axis=1), axis=0)
    R_dot = np.array([[0, 0, 0, 0],
                      [J_beta_11, 0, 0, J_beta_12],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [J_beta_21, 0, 0, J_beta_22]])

    return R_dot


def calculate_M_bar(physical_parameters: dict, q: ArrayLike, q_dot: ArrayLike
) -> ArrayLike:
    M = calculate_M(physical_parameters, q)
    J_beta = calculate_J_beta(physical_parameters, q)

    m_11 = M[0, 0]
    m_12 = M[0, 1]
    m_13 = M[0, 2]
    m_14 = M[0, 3]
    m_22 = M[1, 1]
    m_23 = M[1, 2]
    m_24 = M[1, 3]
    m_33 = M[2, 2]
    m_34 = M[2, 3]
    m_44 = M[3, 3]
    m_55 = M[4, 4]
    m_56 = M[4, 5]
    m_66 = M[5, 5]

    J_beta_11 = J_beta[0, 0]
    J_beta_12 = J_beta[0, 1]
    J_beta_21 = J_beta[1, 0]
    J_beta_22 = J_beta[1, 1]

    m_r11 = m_66 * J_beta_21 ** 2 + m_22 * J_beta_11 ** 2 + 2 * m_12 * J_beta_11 + m_11
    m_r12 = m_13 + J_beta_11 * m_23
    m_r13 = m_14 + J_beta_11 * m_24
    m_r14 = J_beta_21 * m_56 + J_beta_12 * (m_12 + J_beta_11 * m_22) + J_beta_21 * J_beta_22 * m_66
    m_r22 = m_33
    m_r23 = m_34
    m_r24 = J_beta_12 * m_23
    m_r33 = m_44
    m_r34 = J_beta_12 * m_24
    m_r44 = m_66 * J_beta_22 ** 2 + m_22 * J_beta_12 ** 2 + 2 * m_56 * J_beta_11 + m_55

    M_bar = np.array([[m_r11, m_r12, m_r13, m_r14],
                      [m_r12, m_r22, m_r23, m_r24],
                      [m_r13, m_r23, m_r33, m_r34],
                      [m_r14, m_r24, m_r34, m_r44]])

    R = calculate_R(physical_parameters, q)
    M_bar_ = R.T @ M @ R

    return M_bar


def calculate_C_bar(physical_parameters: dict, q: ArrayLike, q_dot: ArrayLike
) -> ArrayLike:
    M = calculate_M(physical_parameters, q)
    C_ = calculate_C(physical_parameters, q, q_dot)
    J_beta = calculate_J_beta(physical_parameters, q)
    J_beta_dot = calculate_J_beta_dot(physical_parameters, q, q_dot)

    m_12 = M[0, 1]
    m_22 = M[1, 1]
    m_23 = M[1, 2]
    m_24 = M[1, 3]
    m_56 = M[4, 5]
    m_66 = M[5, 5]

    c_11 = C_[0, 0]
    c_12 = C_[0, 1]
    c_13 = C_[0, 2]
    c_14 = C_[0, 3]
    c_21 = C_[1, 0]
    c_22 = C_[1, 1]
    c_23 = C_[1, 2]
    c_24 = C_[1, 3]
    c_31 = C_[2, 0]
    c_32 = C_[2, 1]
    c_33 = C_[2, 2]
    c_34 = C_[2, 3]
    c_41 = C_[3, 0]
    c_42 = C_[3, 1]
    c_43 = C_[3, 2]
    c_44 = C_[3, 3]
    c_55 = C_[4, 4]
    c_56 = C_[4, 5]
    c_65 = C_[5, 4]
    c_66 = C_[5, 5]

    J_beta_11 = J_beta[0, 0]
    J_beta_12 = J_beta[0, 1]
    J_beta_21 = J_beta[1, 0]
    J_beta_22 = J_beta[1, 1]

    J_beta_11_dot = J_beta_dot[0, 0]
    J_beta_12_dot = J_beta_dot[0, 1]
    J_beta_21_dot = J_beta_dot[1, 0]
    J_beta_22_dot = J_beta_dot[1, 1]

    c_r11 = (c_66 * J_beta_21 ** 2 + J_beta_21_dot * m_66 * J_beta_21 + c_11 + c_21 * J_beta_11
             + J_beta_11 * (c_12 + c_22 * J_beta_11) + J_beta_11_dot * (m_12 + J_beta_11 * m_22))
    c_r12 = c_13 + c_23 * J_beta_11
    c_r13 = c_14 + c_24 * J_beta_11
    c_r14 = (J_beta_12 * (c_12 + c_22 * J_beta_11) + J_beta_12_dot * (m_12 + J_beta_11 * m_22)
             + c_65 * J_beta_21 + c_66 * J_beta_21 * J_beta_22 + J_beta_21 * J_beta_22_dot * m_66)
    c_r21 = c_31 + J_beta_11_dot * m_23 + c_32 * J_beta_11
    c_r22 = c_33
    c_r23 = c_34
    c_r24 = J_beta_12_dot * m_23 + c_32 * J_beta_12
    c_r31 = c_41 + J_beta_11_dot * m_24 + c_42 * J_beta_11
    c_r32 = c_43
    c_r33 = c_44
    c_r34 = J_beta_12_dot * m_24 + c_42 * J_beta_12
    c_r41 = (J_beta_21 * (c_56 + c_66 * J_beta_22) + J_beta_21_dot * (m_56 + J_beta_22 * m_66)
             + c_21 * J_beta_12 + c_22 * J_beta_11 * J_beta_12 + J_beta_12 * J_beta_11_dot * m_22)
    c_r42 = c_23 * J_beta_12
    c_r43 = c_24 * J_beta_12
    c_r44 = (c_22 * J_beta_12 ** 2 + J_beta_12_dot * m_22 * J_beta_12 + c_55 + c_65 * J_beta_22
             + J_beta_22 * (c_56 + c_66 * J_beta_22) + J_beta_22_dot * (m_56 + J_beta_22 * m_66))

    C_bar = np.array([[c_r11, c_r12, c_r13, c_r14],
                      [c_r21, c_r22, c_r23, c_r24],
                      [c_r31, c_r32, c_r33, c_r34],
                      [c_r41, c_r42, c_r43, c_r44]])

    R = calculate_R(physical_parameters, q)
    R_dot = calculate_R_dot(physical_parameters, q, q_dot)
    C_bar_ = R.T @ C_ @ R + R.T @ M @ R_dot

    return C_bar


def calculate_g_bar(physical_parameters: dict, q: ArrayLike
) -> ArrayLike:
    g_bar = calculate_g(physical_parameters, q)

    R = calculate_R(physical_parameters, q)

    return g_bar


def calculate_tau_bar(physical_parameters: dict, tau: ArrayLike, q: ArrayLike
) -> ArrayLike:
    # R = calculate_R(physical_parameters, q)
    # tau_bar = R.T @ tau

    tau_bar = tau

    return tau_bar


def initialize_physical_parameters(

) -> Dict[str, ArrayLike]:
    I_1 = 0.0017        # [kg m^2]
    I_2 = 0.00001       # [kg m^2]
    I_3 = 0.00001       # [kg m^2]
    I_4 = 0.0014        # [kg m^2]
    I_px = 0.0017       # [kg m^2]
    I_pz = 0.000003     # [kg m^2]

    m_1 = 0.126         # [kg]
    m_2 = 0.08535       # [kg]
    m_3 = 0.063         # [kg]
    m_4 = 0.121         # [kg]
    m_p = 0.125         # [kg]

    l_c1 = 0.047        # [m]
    l_c2 = 0.069        # [m]
    l_c3 = 0.062        # [m]
    l_c4 = 0.045        # [m]
    l_cp = 0.155        # [m]
    L = 0.127           # [m]

    inertias = np.array([I_1, I_2, I_3, I_4, I_px, I_pz])
    masses = np.array([m_1, m_2, m_3, m_4, m_p])
    lengths = np.array([l_c1, l_c2, l_c3, l_c4, l_cp, L])

    physical_parameters = {'inertias': inertias, 'masses': masses, 'lengths': lengths}

    return physical_parameters


def initialize_q(q_1_0, phi_0, theta_0, q_2_0) -> ArrayLike:
    q_0 = np.array([q_1_0, phi_0, theta_0, q_2_0])
    return q_0


def initialize_q_dot(q_1_dot_0, phi_dot_0, theta_dot_0, q_2_dot_0) -> ArrayLike:
    q_dot_0 = np.array([q_1_dot_0, phi_dot_0, theta_dot_0, q_2_dot_0])
    return q_dot_0


def calculate_link_ends(physical_parameters: dict, q: ArrayLike, q_dot: ArrayLike
) -> ArrayLike:
    lengths = physical_parameters['lengths']
    L = lengths[5]

    q_1 = q[0]
    phi = q[1]
    theta = q[2]
    q_2 = q[3]

    beta_1, beta_2, x_b, y_b = direct_kinematics(physical_parameters, q)
    point_P_b = L * np.array([S(theta), -C(theta) * S(phi), C(theta) * C(phi)])
    R_b_0 = np.array([[C(q_1 + beta_1), -S(q_1 + beta_1), 0],
                      [S(q_1 + beta_1),  C(q_1 + beta_1), 0],
                      [0,                0,               1]])

    point_A = np.zeros((3))
    point_B = np.array([L * C(q_1), L * S(q_1), 0])
    point_C = point_B + np.array([L * C(q_1 + beta_1), L * S(q_1 + beta_1), 0])
    # point_C = [x_b, y_b, 0]
    point_D = np.array([2*L + L * C(q_2), L * S(q_2), 0])
    point_E = np.array([2*L, 0, 0])
    point_P = point_C + R_b_0 @ point_P_b

    link_1_ends = np.array([point_A, point_B])
    link_2_ends = np.array([point_B, point_C])
    link_3_ends = np.array([point_C, point_D])
    link_4_ends = np.array([point_D, point_E])
    link_p_ends = np.array([point_C, point_P])

    link_ends = np.array([link_1_ends, link_2_ends, link_3_ends, link_4_ends, link_p_ends])

    return link_ends





def plot_mechanism_3d(link_ends):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot links
    for i, link in enumerate(link_ends):
        ax.plot([link[0][0], link[1][0]],
                [link[0][1], link[1][1]],
                [link[0][2], link[1][2]], color='b')

    # Plot end points
    joints = np.unique(np.reshape(link_ends, (-1, 3)), axis=0)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

    # Set labels and title
    ax.set_xlabel('x_0')
    ax.set_ylabel('y_0')
    ax.set_zlabel('z_0')
    ax.set_title('SIP+FBM')

    plt.show()


def update(frame, link_ends_list, lines, points):
    link_ends_list_current = link_ends_list[frame]

    for i, link in enumerate(link_ends_list_current):
        # lines[i].set_data([link[0][0], link[1][0]],
        #                   [link[0][1], link[1][1]])
        # lines[i].set_3d_properties([link[0][2], link[1][2]])
        lines[i].set_data_3d([link[0][0], link[1][0]],
                             [link[0][1], link[1][1]],
                             [link[0][2], link[1][2]])

    joints = np.unique(np.reshape(link_ends_list_current, (-1, 3)), axis=0)
    points.set_offsets(np.c_[joints[:, 0], joints[:, 1]])
    points.set_3d_properties(joints[:, 2], 'z')

    return lines, points


def animate_mechanism_3d(link_ends_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    link_ends_start = link_ends_list[0]

    num_links = len(link_ends_start)

    lines = []

    # Plot links
    for i, link in enumerate(link_ends_start):
        line, = ax.plot([link[0][0], link[1][0]],
                        [link[0][1], link[1][1]],
                        [link[0][2], link[1][2]], color='b')
        lines.append(line)

    # Plot end points
    joints = np.unique(np.reshape(link_ends_start, (-1, 3)), axis=0)
    points = ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

    # Set labels and title
    ax.set_xlabel('x_0')
    ax.set_ylabel('y_0')
    ax.set_zlabel('z_0')
    ax.set_title('SIP+FBM')
    ax.set_xlim(-0.1, 0.4)
    ax.set_ylim(0, 0.4)
    ax.set_zlim(-0.15, 0.15)
    ax.set_aspect('equal', adjustable='box')

    anim = FuncAnimation(fig, update, frames=len(link_ends_list),
                         fargs=(link_ends_list, lines, points), interval=1)

    plt.show()



def calc_row(q):
    raise NotImplementedError


def calc_drow(q, dq):
    raise NotImplementedError
