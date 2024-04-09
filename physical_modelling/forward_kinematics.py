import numpy as np

from functools import partial
from typing import Tuple, Dict, Any
from numpy.typing import NDArray


# calculate beta_1 and beta_2 from q_1 and q_2 + x_b and y_b (pendulum base positions)
def reduced_forward_kinematics(physical_parameters: Dict, q: NDArray
) -> Tuple[float, float, float, float]:
    L = physical_parameters['L']

    q_1 = q[0]
    q_2 = q[3]

    # TODO: generalize this for all angles (may not be necessary with state constraints and too much work)

    x_B = L * np.cos(q_1)
    y_B = L * np.sin(q_1)
    x_D = 2*L + L * np.cos(q_2)
    y_D = L * np.sin(q_2)
    p_B = np.array([x_B, y_B])
    p_D = np.array([x_D, y_D])
    if np.linalg.norm(p_B - p_D) > 2*L:
        # configuration not possible
        return 0, 0, 0, 0

    fraction_aux = np.cos(q_1 - q_2) + 2 * np.cos(q_1) - 2 * np.cos(q_2)
    sqrt_term = np.sqrt(-(fraction_aux - 1) / (fraction_aux - 3))

    x_b_1 = L / 2 * (2 + np.cos(q_1) + np.cos(q_2)) + L / 2 * (np.sin(q_1) - np.sin(q_2)) * sqrt_term
    x_b_2 = L / 2 * (2 + np.cos(q_1) + np.cos(q_2)) - L / 2 * (np.sin(q_1) - np.sin(q_2)) * sqrt_term

    y_b_1 = L / 2 * (np.sin(q_1) + np.sin(q_2)) + L / 2 * (np.cos(q_2) - np.cos(q_1) + 2) * sqrt_term
    y_b_2 = L / 2 * (np.sin(q_1) + np.sin(q_2)) - L / 2 * (np.cos(q_2) - np.cos(q_1) + 2) * sqrt_term

    # select correct x_b, y_b (with proper state constraints this should be true, i.e. close to the equilibrium point)
    if np.abs(q_2 - q_1) <= np.pi:
        x_b = x_b_1
        y_b = y_b_1
    else:
        x_b = x_b_2
        y_b = y_b_2

    # incorrect formula in the paper
    beta_1 = np.arctan2(y_b - L * np.sin(q_1), x_b - L * np.cos(q_1)) - q_1
    beta_2 = np.arctan2(y_b - L * np.sin(q_2), x_b - (2 * L + L * np.cos(q_2))) - q_2

    return beta_1, beta_2, x_b, y_b


# differential kinematics
def extended_reduced_forward_kinematics(physical_parameters: Dict, q: NDArray, q_dot: NDArray
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    from .jacobians import jacobian, jacobian_beta

    q_1_dot = q_dot[0]
    q_2_dot = q_dot[3]

    # velocity of the pendulum's base
    J = jacobian(physical_parameters, q)
    x_dot = J @ q_dot
    x_b_dot = x_dot[0]
    y_b_dot = x_dot[1]

    # relative velocities
    J_beta = jacobian_beta(physical_parameters, q)
    q_dot_reduced = np.array([q_1_dot, q_2_dot])
    beta_dot = J_beta @ q_dot_reduced
    beta_1_dot = beta_dot[0]
    beta_2_dot = beta_dot[1]

    return beta_1_dot, beta_2_dot, x_b_dot, y_b_dot


def extended_reduced_forward_kinematics_with_accelerations(physical_parameters: Dict, q: NDArray, q_dot: NDArray
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    from .jacobians import jacobian, jacobian_beta, jacobian_dot, jacobian_beta_dot
    from .dynamics import continuous_forward_dynamics

    beta_1_dot, beta_2_dot, x_b_dot, y_b_dot = extended_reduced_forward_kinematics(physical_parameters, q, q_dot)

    q_1_dot = q_dot[0]
    q_2_dot = q_dot[3]

    q_ddot = continuous_forward_dynamics(physical_parameters, q, q_dot)
    q_1_ddot = q_ddot[0]
    q_2_ddot = q_ddot[3]

    # acceleration of the pendulum's base
    J = jacobian(physical_parameters, q)
    J_dot = jacobian_dot(physical_parameters, q, q_dot)
    x_ddot = J @ q_ddot + J_dot @ q_dot
    x_b_ddot = x_ddot[0]
    y_b_ddot = x_ddot[1]

    # relative accelerations
    J_beta = jacobian_beta(physical_parameters, q)
    J_beta_dot = jacobian_beta_dot(physical_parameters, q, q_dot)
    q_dot_reduced = np.array([q_1_dot, q_2_dot])
    q_ddot_reduced = np.array([q_1_ddot, q_2_ddot])
    beta_ddot = J_beta @ q_ddot_reduced + J_beta_dot @ q_dot_reduced
    beta_1_ddot = beta_ddot[0]
    beta_2_ddot = beta_ddot[1]

    return beta_1_dot, beta_2_dot, x_b_dot, y_b_dot, beta_1_ddot, beta_2_ddot, x_b_ddot, y_b_ddot


def forward_kinematics(physical_parameters: Dict, q: NDArray
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    L = physical_parameters['L']

    q_1 = q[0]
    phi = q[1]
    theta = q[2]
    q_2 = q[3]

    beta_1, beta_2, x_b, y_b = reduced_forward_kinematics(physical_parameters, q)

    # link endpoints
    point_A = np.zeros(3)
    point_B = np.array([L * np.cos(q_1), L * np.sin(q_1), 0])
    # point_C = point_B + np.array([L * np.cos(q_1 + beta_1), L * np.sin(q_1 + beta_1), 0])
    point_C = np.array([x_b, y_b, 0])
    point_D = np.array([2 * L + L * np.cos(q_2), L * np.sin(q_2), 0])
    point_E = np.array([2 * L, 0, 0])

    # pendulum endpoint
    point_P_b = L * np.array([np.sin(theta), -np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)])
    R_b_0 = np.array([[np.cos(q_1 + beta_1), -np.sin(q_1 + beta_1), 0],
                      [np.sin(q_1 + beta_1),  np.cos(q_1 + beta_1), 0],
                      [0,                     0,                    1]])
    point_P = point_C + R_b_0 @ point_P_b

    return point_A, point_B, point_C, point_D, point_E, point_P


# differential kinematics
def extended_forward_kinematics(physical_parameters: Dict, q: NDArray, q_dot: NDArray, q_ddot: NDArray
) -> Dict[str, Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]]:
    from .jacobians import jacobian, jacobian_dot

    L = physical_parameters['L']

    q_1 = q[0]
    phi = q[1]
    theta = q[2]
    q_2 = q[3]
    q_1_dot = q_dot[0]
    phi_dot = q_dot[1]
    theta_dot = q_dot[2]
    q_2_dot = q_dot[3]
    q_1_ddot = q_ddot[0]
    phi_ddot = q_ddot[1]
    theta_ddot = q_ddot[2]
    q_2_ddot = q_ddot[3]

    beta_1, beta_2, _, _ = reduced_forward_kinematics(physical_parameters, q)
    x_A, x_B, x_C, x_D, x_E, x_P = forward_kinematics(physical_parameters, q)
    beta_1_dot, beta_2_dot, x_b_dot, y_b_dot, beta_1_ddot, beta_2_ddot, x_b_ddot, y_b_ddot \
        = extended_reduced_forward_kinematics_with_accelerations(physical_parameters, q, q_dot)

    J = jacobian(physical_parameters, q)
    J_dot = jacobian_dot(physical_parameters, q, q_dot)

    # velocities
    x_A_dot = np.zeros(3)
    x_B_dot = np.array([- L * np.sin(q_1), L * np.cos(q_1), 0]) * q_1_dot
    x_C_dot = np.array([x_b_dot, y_b_dot, 0])
    x_D_dot = np.array([- L * np.sin(q_2), L * np.cos(q_2), 0]) * q_2_dot
    x_E_dot = np.zeros(3)
    x_P_dot = x_C_dot + L * np.array([- np.sin(q_1 + beta_1) * np.sin(theta) * (q_1_dot + beta_1_dot)
                                      + np.cos(q_1 + beta_1) * np.cos(theta) * theta_dot
                                      + np.cos(q_1 + beta_1) * np.cos(theta) * np.sin(phi) * (q_1_dot + beta_1_dot)
                                      - np.sin(q_1 + beta_1) * np.sin(theta) * np.sin(phi) * theta_dot
                                      + np.sin(q_1 + beta_1) * np.cos(theta) * np.cos(phi) * phi_dot,
                                      + np.cos(q_1 + beta_1) * np.sin(theta) * (q_1_dot + beta_1_dot)
                                      + np.sin(q_1 + beta_1) * np.cos(theta) * theta_dot
                                      + np.sin(q_1 + beta_1) * np.cos(theta) * np.sin(phi)* (q_1_dot + beta_1_dot)
                                      + np.cos(q_1 + beta_1) * np.sin(theta) * np.sin(phi)* theta_dot
                                      - np.cos(q_1 + beta_1) * np.cos(theta) * np.cos(phi)* phi_dot,
                                      - np.sin(theta) * np.cos(phi) * theta_dot
                                      - np.cos(theta) * np.sin(phi) * phi_dot])

    # accelerations
    x_ddot_ = J @ q_ddot + J_dot @ q_dot
    x_b_ddot = x_ddot_[0]
    y_b_ddot = x_ddot_[1]

    x_A_ddot = np.zeros(3)
    x_B_ddot = np.array([- L * np.cos(q_1) * q_1_dot ** 2 - L * np.sin(q_1) * q_1_ddot,
                         - L * np.sin(q_1) * q_1_dot ** 2 + L * np.cos(q_1) * q_1_ddot,
                         0])
    x_C_ddot = np.array([x_b_ddot, y_b_ddot, 0])
    x_D_ddot = np.array([- L * np.cos(q_2) * q_2_dot ** 2 - L * np.sin(q_2) * q_2_ddot,
                         - L * np.sin(q_2) * q_2_dot ** 2 + L * np.cos(q_2) * q_2_ddot,
                         0])
    x_E_ddot = np.zeros(3)
    x_P_ddot = x_C_ddot + L * np.array([+ phi_ddot*np.sin(beta_1 + q_1)*np.cos(phi)*np.cos(theta)
                                        - phi_dot**2*np.sin(beta_1 + q_1)*np.sin(phi)*np.cos(theta)
                                        - 2*phi_dot*theta_dot*np.sin(beta_1 + q_1)*np.sin(theta)*np.cos(phi)
                                        + 2*phi_dot*(beta_1_dot + q_1_dot)*np.cos(beta_1 + q_1)*np.cos(phi)*np.cos(theta)
                                        - theta_ddot*np.sin(beta_1 + q_1)*np.sin(phi)*np.sin(theta)
                                        + theta_ddot*np.cos(beta_1 + q_1)*np.cos(theta)
                                        - theta_dot**2*np.sin(beta_1 + q_1)*np.sin(phi)*np.cos(theta)
                                        - theta_dot**2*np.sin(theta)*np.cos(beta_1 + q_1)
                                        - 2*theta_dot*(beta_1_dot + q_1_dot)*np.sin(beta_1 + q_1)*np.cos(theta)
                                        - 2*theta_dot*(beta_1_dot + q_1_dot)*np.sin(phi)*np.sin(theta)*np.cos(beta_1 + q_1)
                                        - (beta_1_ddot + q_1_ddot)*np.sin(beta_1 + q_1)*np.sin(theta)
                                        + (beta_1_ddot + q_1_ddot)*np.sin(phi)*np.cos(beta_1 + q_1)*np.cos(theta)
                                        - (beta_1_dot + q_1_dot)**2*np.sin(beta_1 + q_1)*np.sin(phi)*np.cos(theta)
                                        - (beta_1_dot + q_1_dot)**2*np.sin(theta)*np.cos(beta_1 + q_1),
                                        - phi_ddot*np.cos(beta_1 + q_1)*np.cos(phi)*np.cos(theta)
                                        + phi_dot**2*np.sin(phi)*np.cos(beta_1 + q_1)*np.cos(theta)
                                        + 2*phi_dot*theta_dot*np.sin(theta)*np.cos(beta_1 + q_1)*np.cos(phi)
                                        + 2*phi_dot*(beta_1_dot + q_1_dot)*np.sin(beta_1 + q_1)*np.cos(phi)*np.cos(theta)
                                        + theta_ddot*np.sin(beta_1 + q_1)*np.cos(theta)
                                        + theta_ddot*np.sin(phi)*np.sin(theta)*np.cos(beta_1 + q_1)
                                        - theta_dot**2*np.sin(beta_1 + q_1)*np.sin(theta)
                                        + theta_dot**2*np.sin(phi)*np.cos(beta_1 + q_1)*np.cos(theta)
                                        - 2*theta_dot*(beta_1_dot + q_1_dot)*np.sin(beta_1 + q_1)*np.sin(phi)*np.sin(theta)
                                        + 2*theta_dot*(beta_1_dot + q_1_dot)*np.cos(beta_1 + q_1)*np.cos(theta)
                                        + (beta_1_ddot + q_1_ddot)*np.sin(beta_1 + q_1)*np.sin(phi)*np.cos(theta)
                                        + (beta_1_ddot + q_1_ddot)*np.sin(theta)*np.cos(beta_1 + q_1)
                                        - (beta_1_dot + q_1_dot)**2*np.sin(beta_1 + q_1)*np.sin(theta)
                                        + (beta_1_dot + q_1_dot)**2*np.sin(phi)*np.cos(beta_1 + q_1)*np.cos(theta),
                                        - phi_ddot*np.sin(phi)*np.cos(theta)
                                        - phi_dot**2*np.cos(phi)*np.cos(theta)
                                        + 2*phi_dot*theta_dot*np.sin(phi)*np.sin(theta)
                                        - theta_ddot*np.sin(theta)*np.cos(phi)
                                        - theta_dot**2*np.cos(phi)*np.cos(theta)])

    kinematics_dict = {'x': forward_kinematics(physical_parameters, q),
                       'x_dot': (x_A_dot, x_B_dot, x_C_dot, x_D_dot, x_E_dot, x_P_dot),
                       'x_ddot': (x_A_ddot, x_B_ddot, x_C_ddot, x_D_ddot, x_E_ddot, x_P_ddot)}

    return kinematics_dict


def calculate_link_endpoints(physical_parameters: dict, q: NDArray
) -> NDArray:
    point_A, point_B, point_C, point_D, point_E, point_P = forward_kinematics(physical_parameters, q)

    link_1_ends = np.array([point_A, point_B])
    link_2_ends = np.array([point_B, point_C])
    link_3_ends = np.array([point_C, point_D])
    link_4_ends = np.array([point_D, point_E])
    link_p_ends = np.array([point_C, point_P])

    link_ends = np.array([link_1_ends, link_2_ends, link_3_ends, link_4_ends, link_p_ends])

    return link_ends
