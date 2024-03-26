import numpy as np

from typing import Dict, Tuple
from numpy.typing import NDArray

from forward_kinematics import reduced_forward_kinematics, extended_reduced_forward_kinematics, forward_kinematics
from jacobians import jacobian_beta, jacobian_beta_dot


def S(angle):
    return np.sin(angle)


def C(angle):
    return np.cos(angle)


def _calculate_M(physical_parameters: dict, q: NDArray
) -> NDArray:
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

    phi = q[1]
    theta = q[2]

    beta_1, beta_2, _, _ = reduced_forward_kinematics(physical_parameters, q)

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


def _calculate_C(physical_parameters: dict, q: NDArray, q_dot: NDArray
) -> NDArray:
    I_px = physical_parameters['I_px']
    I_pz = physical_parameters['I_pz']
    m_2 = physical_parameters['m_2']
    m_3 = physical_parameters['m_3']
    m_p = physical_parameters['m_p']
    l_c2 = physical_parameters['l_c2']
    l_c3 = physical_parameters['l_c3']
    l_cp = physical_parameters['l_cp']
    L = physical_parameters['L']

    phi = q[1]
    theta = q[2]

    q_1_dot = q_dot[0]
    phi_dot = q_dot[1]
    theta_dot = q_dot[2]
    q_2_dot = q_dot[3]

    beta_1, beta_2, _, _ = reduced_forward_kinematics(physical_parameters, q)
    beta_1_dot, beta_2_dot, _, _ = extended_reduced_forward_kinematics(physical_parameters, q, q_dot)

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


def _calculate_g(physical_parameters: dict, q: NDArray
) -> NDArray:
    m_p = physical_parameters['mp']
    l_cp = physical_parameters['l_cp']

    phi = q[1]
    theta = q[2]

    _g = 9.81

    g = - m_p * _g * l_cp * np.array([0, S(phi) * C(theta), C(phi) * S(theta), 0])

    return g


def _calculate_R(physical_parameters: dict, q: NDArray
) -> NDArray:
    J_beta = jacobian_beta(physical_parameters, q)

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


def _calculate_R_dot(physical_parameters: dict, q: NDArray, q_dot: NDArray
) -> NDArray:
    J_beta_dot = jacobian_beta_dot(physical_parameters, q, q_dot)

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


def _calculate_M_bar(physical_parameters: dict, q: NDArray
) -> NDArray:
    M = _calculate_M(physical_parameters, q)

    R = _calculate_R(physical_parameters, q)
    M_bar_ = R.T @ M @ R

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

    J_beta = jacobian_beta(physical_parameters, q)
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

    return M_bar_


def _calculate_C_bar(physical_parameters: dict, q: NDArray, q_dot: NDArray
) -> NDArray:
    M = _calculate_M(physical_parameters, q)
    C_ = _calculate_C(physical_parameters, q, q_dot)

    R = _calculate_R(physical_parameters, q)
    R_dot = _calculate_R_dot(physical_parameters, q, q_dot)
    C_bar_ = R.T @ C_ @ R + R.T @ M @ R_dot

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

    J_beta = jacobian_beta(physical_parameters, q)
    J_beta_11 = J_beta[0, 0]
    J_beta_12 = J_beta[0, 1]
    J_beta_21 = J_beta[1, 0]
    J_beta_22 = J_beta[1, 1]

    J_beta_dot = jacobian_beta_dot(physical_parameters, q, q_dot)
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


    return C_bar_


def _calculate_g_bar(physical_parameters: dict, q: NDArray
) -> NDArray:
    g_bar = _calculate_g(physical_parameters, q)

    R = _calculate_R(physical_parameters, q)

    return g_bar


def dynamical_matrices(physical_parameters: dict, q: NDArray, q_dot: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    M = _calculate_M(physical_parameters, q)
    C_ = _calculate_C(physical_parameters, q, q_dot)
    g = _calculate_g(physical_parameters, q)
    R = _calculate_R(physical_parameters, q)
    R_dot = _calculate_R_dot(physical_parameters, q, q_dot)

    M_bar = R.T @ M @ R
    C_bar = R.T @ C_ @ R + R.T @ M @ R_dot
    g_bar = g

    return M_bar_, C_bar, g_bar


def calculate_tau_bar(physical_parameters: dict, tau: NDArray, q: NDArray
) -> NDArray:
    # R = calculate_R(physical_parameters, q)
    # tau_bar = R.T @ tau

    tau_bar = tau

    return tau_bar