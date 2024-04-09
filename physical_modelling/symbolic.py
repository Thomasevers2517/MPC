import sympy as sp

from typing import List, Tuple, Dict
from time import time

from physical_modelling.initialisation import initialize_physical_parameters


def rotation_matrix(axis: str, angle) -> sp.Matrix:
    C = sp.cos(angle)
    S = sp.sin(angle)

    _R_x = sp.Matrix([[ 1,  0,  0],
                      [ 0,  C, -S],
                      [ 0,  S,  C]])
    _R_y = sp.Matrix([[ C,  0,  S],
                      [ 0,  1,  0],
                      [-S,  0,  C]])
    _R_z = sp.Matrix([[ C, -S,  0],
                      [ S,  C,  0],
                      [ 0,  0,  1]])

    assert axis.lower() in ['x', 'y', 'z'], "Invalid axis"

    if axis in ['x', 'X']:
        return _R_x

    if axis in ['y', 'Y']:
        return _R_y

    if axis in ['z', 'Z']:
        return _R_z


def define_symbols() -> Tuple[sp.Symbol, Dict[str, sp.Symbol], sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix,
                              List[sp.Symbol], List[sp.Symbol]]:
    t = sp.symbols('t')
    I_1, I_2, I_3, I_4, I_px, I_pz, m_1, m_2, m_3, m_4, m_p, l_c1, l_c2, l_c3, l_c4, l_cp, L, g_0 \
        = sp.symbols('I_1, I_2, I_3, I_4, I_px, I_pz, m_1, m_2, m_3, m_4, m_p, l_c1, l_c2, l_c3, l_c4, l_cp, L, g_0')
    q_1_dot_, phi_dot_, theta_dot_, q_2_dot_, q_1_ddot_, phi_ddot_, theta_ddot_, q_2_ddot_ \
        = sp.symbols('q_1_dot, phi_dot, theta_dot, q_2_dot, q_1_ddot, phi_ddot, theta_ddot, q_2_ddot')
    beta_1_dot_, beta_2_dot_, beta_1_ddot_, beta_2_ddot_ \
        = sp.symbols('beta_1_dot, beta_2_dot, beta_1_ddot, beta_2_ddot')

    q_1 = sp.Function('q_1')
    phi = sp.Function('phi')
    theta = sp.Function('theta')
    q_2 = sp.Function('q_2')
    beta_1 = sp.Function('beta_1')
    beta_2 = sp.Function('beta_2')

    q_1_dot = sp.Derivative(q_1(t), t)
    phi_dot = sp.Derivative(phi(t), t)
    theta_dot = sp.Derivative(theta(t), t)
    q_2_dot = sp.Derivative(q_2(t), t)
    q_1_ddot = sp.Derivative(q_1(t), t, 2)
    phi_ddot = sp.Derivative(phi(t), t, 2)
    theta_ddot = sp.Derivative(theta(t), t, 2)
    q_2_ddot = sp.Derivative(q_2(t), t, 2)

    beta_1_dot = sp.Derivative(beta_1(t), t)
    beta_2_dot = sp.Derivative(beta_2(t), t)
    beta_1_ddot = sp.Derivative(beta_1(t), t, 2)
    beta_2_ddot = sp.Derivative(beta_2(t), t, 2)

    parameters = dict(I_1=I_1, I_2=I_2, I_3=I_3, I_4=I_4, I_px=I_px, I_pz=I_pz,
                      m_1=m_1, m_2=m_2, m_3=m_3, m_4=m_4, m_p=m_p,
                      l_c1=l_c1, l_c2=l_c2, l_c3=l_c3, l_c4=l_c4, l_cp=l_cp, L=L,
                      g_0=g_0)

    q = sp.Matrix([q_1(t),
                   phi(t),
                   theta(t),
                   q_2(t)])
    q_dot = sp.Matrix([q_1_dot,
                       phi_dot,
                       theta_dot,
                       q_2_dot])

    rho = sp.Matrix([q_1(t),
                     beta_1(t),
                     phi(t),
                     theta(t),
                     q_2(t),
                     beta_2(t)])
    rho_dot = sp.Matrix([q_1_dot,
                         beta_1_dot,
                         phi_dot,
                         theta_dot,
                         q_2_dot,
                         beta_2_dot])

    substitution_old = [q_1_dot, phi_dot, theta_dot, q_2_dot, q_1_ddot, phi_ddot, theta_ddot, q_2_ddot,
                        beta_1_dot, beta_2_dot, beta_1_ddot, beta_2_ddot]
    substitution_new = [q_1_dot_, phi_dot_, theta_dot_, q_2_dot_, q_1_ddot_, phi_ddot_, theta_ddot_, q_2_ddot_,
                        beta_1_dot_, beta_2_dot_, beta_1_ddot_, beta_2_ddot_]

    return t, parameters, q, q_dot, rho, rho_dot, substitution_old, substitution_new


def simplify_and_substitute(expression, old: List = None, new: List = None):
    _, _, _, _, _, _, substitution_old, substitution_new = define_symbols()
    if old is None:
        old = substitution_old
    if new is None:
        new = substitution_new

    assert len(old) == len(new), "Number of old and new variables must match"

    subs_dict = dict(zip(old, new))

    # subbed = expression.subs(subs_dict)
    # subbed_and_simplified = sp.simplify(subbed)

    simplified = sp.simplify(expression)
    simplified_and_subbed = simplified.subs(subs_dict)

    result = simplified_and_subbed

    return result


def numeric_substitution(expression, numerical_values):
    _, _, _, _, _, _, substitution_old, substitution_new = define_symbols()
    numerical_parameters = initialize_physical_parameters()

    subbed = expression.xreplace(dict(zip(substitution_old, substitution_new)))
    subbed = subbed.xreplace(numerical_parameters)
    subbed = subbed.xreplace(dict(zip(substitution_new, numerical_values)))
    subbed = sp.nsimplify(subbed)

    return subbed


def symbolic_forward_kinematics_positions(CoM_or_joint: str) -> List[sp.Matrix]:
    t, parameters, _, _, rho, _, _, _ = define_symbols()

    l_c1 = parameters['l_c1']
    l_c2 = parameters['l_c2']
    l_c3 = parameters['l_c3']
    l_c4 = parameters['l_c4']
    l_cp = parameters['l_cp']
    L = parameters['L']

    q_1 = rho[0]
    beta_1 = rho[1]
    phi = rho[2]
    theta = rho[3]
    q_2 = rho[4]
    beta_2 = rho[5]

    r_P_b_b = L * sp.Matrix([sp.sin(theta),
                            -sp.cos(theta) * sp.sin(phi),
                             sp.cos(theta) * sp.cos(phi)])
    r_CoM_b_b = l_cp * sp.Matrix([sp.sin(theta),
                                  -sp.cos(theta) * sp.sin(phi),
                                  sp.cos(theta) * sp.cos(phi)])
    R_b_0 = rotation_matrix('z', q_1 + beta_1)

    # r_P_b_0 = R_b_0 * r_P_b_b

    r_A_0 = sp.zeros(3, 1)
    r_B_0 = sp.Matrix([L * sp.cos(q_1),
                       L * sp.sin(q_1),
                       0])
    r_C_0 = sp.Matrix([L * sp.cos(q_1) + L * sp.cos(q_1 + beta_1),
                       L * sp.sin(q_1) + L * sp.sin(q_1 + beta_1),
                       0])
    r_D_0 = sp.Matrix([2 * L + L * sp.cos(q_2),
                       L * sp.sin(q_2),
                       0])
    r_E_0 = sp.Matrix([2 * L,
                       0,
                       0])
    r_P_0 = r_C_0 + R_b_0 * r_P_b_b
    r_joints = [r_A_0, r_B_0, r_C_0, r_D_0, r_E_0, r_P_0]

    r_CoM_1_0 = sp.Matrix([l_c1 * sp.cos(q_1),
                           l_c1 * sp.sin(q_1),
                           0])
    r_CoM_2_0 = sp.Matrix([L * sp.cos(q_1) + l_c2 * sp.cos(q_1 + beta_1),
                           L * sp.sin(q_1) + l_c2 * sp.sin(q_1 + beta_1),
                           0])
    r_CoM_3_0 = sp.Matrix([2 * L + L * sp.cos(q_2) + l_c3 * sp.cos(q_2 + beta_2),
                           L * sp.sin(q_2) + l_c3 * sp.sin(q_2 + beta_2),
                           0])
    r_CoM_4_0 = sp.Matrix([2 * L + l_c4 * sp.cos(q_2),
                           l_c4 * sp.sin(q_2),
                           0])
    r_CoM_P_0 = r_C_0 + R_b_0 * r_CoM_b_b
    r_CoM = [r_CoM_1_0, r_CoM_2_0, r_CoM_3_0, r_CoM_4_0, r_CoM_P_0]

    assert CoM_or_joint.lower() in ['com', 'joint'], "Can only return CoM or joint positions"

    if CoM_or_joint.lower() == 'com':
        return r_CoM

    if CoM_or_joint.lower() == 'joint':
        return r_joints


def symbolic_forward_kinematics_velocities_and_accelerations(CoM_or_joint: str = 'CoM'
) -> Tuple[List[sp.Matrix], List[sp.Matrix], List[sp.Matrix]]:
    t, _, _, _, rho, rho_dot, _, _ = define_symbols()

    q_1 = rho[0]
    beta_1 = rho[1]
    phi = rho[2]
    theta = rho[3]

    q_1_dot = rho_dot[0]
    beta_1_dot = rho_dot[1]
    q_2_dot = rho_dot[4]
    beta_2_dot = rho_dot[5]

    r_CoM = symbolic_forward_kinematics_positions('CoM')
    r_joint = symbolic_forward_kinematics_positions('joint')

    r_CoM_1_0 = r_CoM[0]
    r_CoM_2_0 = r_CoM[1]
    r_CoM_3_0 = r_CoM[2]
    r_CoM_4_0 = r_CoM[3]
    r_CoM_P_0 = r_CoM[4]

    r_A_0 = r_joint[0]
    r_B_0 = r_joint[1]
    r_C_0 = r_joint[2]
    r_D_0 = r_joint[3]
    r_E_0 = r_joint[4]
    r_P_0 = r_joint[5]

    r_A_dot_0 = simplify_and_substitute(sp.diff(r_A_0, t))
    r_B_dot_0 = simplify_and_substitute(sp.diff(r_B_0, t))
    r_C_dot_0 = simplify_and_substitute(sp.diff(r_C_0, t))
    r_D_dot_0 = simplify_and_substitute(sp.diff(r_D_0, t))
    r_E_dot_0 = simplify_and_substitute(sp.diff(r_E_0, t))
    r_P_dot_0 = simplify_and_substitute(sp.diff(r_P_0, t))
    r_joints_dot = [r_A_dot_0, r_B_dot_0, r_C_dot_0, r_D_dot_0, r_E_dot_0, r_P_dot_0]

    r_CoM_1_dot_0 = simplify_and_substitute(sp.diff(r_CoM_1_0, t))
    r_CoM_2_dot_0 = simplify_and_substitute(sp.diff(r_CoM_2_0, t))
    r_CoM_3_dot_0 = simplify_and_substitute(sp.diff(r_CoM_3_0, t))
    r_CoM_4_dot_0 = simplify_and_substitute(sp.diff(r_CoM_4_0, t))
    r_CoM_P_dot_0 = simplify_and_substitute(sp.diff(r_CoM_P_0, t))
    r_CoM_dot = [r_CoM_1_dot_0, r_CoM_2_dot_0, r_CoM_3_dot_0, r_CoM_4_dot_0, r_CoM_P_dot_0]

    r_A_ddot_0 = simplify_and_substitute(sp.diff(r_A_0, t, 2))
    r_B_ddot_0 = simplify_and_substitute(sp.diff(r_B_0, t, 2))
    r_C_ddot_0 = simplify_and_substitute(sp.diff(r_C_0, t, 2))
    r_D_ddot_0 = simplify_and_substitute(sp.diff(r_D_0, t, 2))
    r_E_ddot_0 = simplify_and_substitute(sp.diff(r_E_0, t, 2))
    r_P_ddot_0 = simplify_and_substitute(sp.diff(r_P_0, t, 2))
    r_joints_ddot = [r_A_ddot_0, r_B_ddot_0, r_C_ddot_0, r_D_ddot_0, r_E_ddot_0, r_P_ddot_0]

    r_CoM_1_ddot_0 = simplify_and_substitute(sp.diff(r_CoM_1_0, t, 2))
    r_CoM_2_ddot_0 = simplify_and_substitute(sp.diff(r_CoM_2_0, t, 2))
    r_CoM_3_ddot_0 = simplify_and_substitute(sp.diff(r_CoM_3_0, t, 2))
    r_CoM_4_ddot_0 = simplify_and_substitute(sp.diff(r_CoM_4_0, t, 2))
    r_CoM_P_ddot_0 = simplify_and_substitute(sp.diff(r_CoM_P_0, t, 2))
    r_CoM_ddot = [r_CoM_1_ddot_0, r_CoM_2_ddot_0, r_CoM_3_ddot_0, r_CoM_4_ddot_0, r_CoM_P_ddot_0]

    R_b_0 = rotation_matrix('z', q_1 + beta_1)
    R_p_b = rotation_matrix('x', phi) * rotation_matrix('y', theta)
    R_p_0 = R_b_0 * R_p_b
    omega_P_0_cross = sp.diff(R_p_0, t) * R_p_0.T

    omega_1_0 = sp.Matrix([0, 0, q_1_dot])
    omega_2_0 = sp.Matrix([0, 0, q_1_dot + beta_1_dot])
    omega_3_0 = sp.Matrix([0, 0, q_2_dot + beta_2_dot])
    omega_4_0 = sp.Matrix([0, 0, q_2_dot])
    omega_P_0 = sp.Matrix([sp.simplify(omega_P_0_cross[2, 1]),
                           sp.simplify(omega_P_0_cross[0, 2]),
                           sp.simplify(omega_P_0_cross[1, 0])])
    omega = [omega_1_0, omega_2_0, omega_3_0, omega_4_0, omega_P_0]

    omega_P_0_cross_simp = simplify_and_substitute(sp.diff(R_p_0, t) * R_p_0.T)
    omega_P_0_simp = sp.Matrix([simplify_and_substitute(omega_P_0_cross_simp[2, 1]),
                                simplify_and_substitute(omega_P_0_cross_simp[0, 2]),
                                simplify_and_substitute(omega_P_0_cross_simp[1, 0])])

    assert CoM_or_joint.lower() in ['com', 'joint'], "Can only return CoM or joint positions"

    if CoM_or_joint.lower() == 'com':
        return r_CoM_dot, r_CoM_ddot, omega

    if CoM_or_joint.lower() == 'end':
        return r_joints_dot, r_joints_ddot, omega


def calculate_inertias() -> List[sp.Matrix]:
    t, parameters, _, _, rho, _, _, _ = define_symbols()

    I_1 = parameters['I_1']
    I_2 = parameters['I_2']
    I_3 = parameters['I_3']
    I_4 = parameters['I_4']
    I_px = parameters['I_px']
    I_pz = parameters['I_pz']

    q_1 = rho[0]
    beta_1 = rho[1]
    phi = rho[2]
    theta = rho[3]
    q_2 = rho[4]
    beta_2 = rho[5]

    R_1_0 = rotation_matrix('z', q_1)
    R_2_0 = rotation_matrix('z', q_1 + beta_1)
    R_3_0 = rotation_matrix('z', q_2 + beta_2)
    R_4_0 = rotation_matrix('z', q_2)
    R_b_0 = rotation_matrix('z', q_1 + beta_1)
    R_p_b = rotation_matrix('x', phi) * rotation_matrix('y', theta)
    R_p_0 = R_b_0 * R_p_b

    I_1_0 = R_1_0 * sp.diag(0, 0, I_1) * R_1_0.T
    I_2_0 = R_2_0 * sp.diag(0, 0, I_2) * R_2_0.T
    I_3_0 = R_3_0 * sp.diag(0, 0, I_3) * R_3_0.T
    I_4_0 = R_4_0 * sp.diag(0, 0, I_4) * R_4_0.T
    I_P_0 = R_p_0 * sp.diag(I_px, I_px, I_pz) * R_p_0.T

    inertias = [I_1_0, I_2_0, I_3_0, I_4_0, I_P_0]

    return inertias


def calculate_jacobians() -> Tuple[List[sp.Matrix], List[sp.Matrix]]:
    t, parameters, _, _, rho, rho_dot, _, _ = define_symbols()

    r_CoM = symbolic_forward_kinematics_positions('CoM')
    r_CoM_1_0 = r_CoM[0]
    r_CoM_2_0 = r_CoM[1]
    r_CoM_3_0 = r_CoM[2]
    r_CoM_4_0 = r_CoM[3]
    r_CoM_P_0 = r_CoM[4]

    _, _, omega = symbolic_forward_kinematics_velocities_and_accelerations()
    omega_1_0 = omega[0]
    omega_2_0 = omega[1]
    omega_3_0 = omega[2]
    omega_4_0 = omega[3]
    omega_P_0 = omega[4]

    J_v_1_0 = r_CoM_1_0.jacobian(rho)
    J_v_2_0 = r_CoM_2_0.jacobian(rho)
    J_v_3_0 = r_CoM_3_0.jacobian(rho)
    J_v_4_0 = r_CoM_4_0.jacobian(rho)
    J_v_P_0 = r_CoM_P_0.jacobian(rho)
    J_v = [J_v_1_0, J_v_2_0, J_v_3_0, J_v_4_0, J_v_P_0]

    J_omega_1_0 = omega_1_0.jacobian(rho_dot)
    J_omega_2_0 = omega_2_0.jacobian(rho_dot)
    J_omega_3_0 = omega_3_0.jacobian(rho_dot)
    J_omega_4_0 = omega_4_0.jacobian(rho_dot)
    J_omega_P_0 = omega_P_0.jacobian(rho_dot)
    J_omega = [J_omega_1_0, J_omega_2_0, J_omega_3_0, J_omega_4_0, J_omega_P_0]

    return J_v, J_omega


def calculate_dynamical_matrices() -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    t, parameters, _, _, rho, rho_dot, _, _ = define_symbols()

    m_1 = parameters['m_1']
    m_2 = parameters['m_2']
    m_3 = parameters['m_3']
    m_4 = parameters['m_4']
    m_p = parameters['m_p']
    g_0 = parameters['g_0']

    J_v, J_omega = calculate_jacobians()
    J_v_1_0 = J_v[0]
    J_v_2_0 = J_v[1]
    J_v_3_0 = J_v[2]
    J_v_4_0 = J_v[3]
    J_v_P_0 = J_v[4]
    J_omega_1_0 = J_omega[0]
    J_omega_2_0 = J_omega[1]
    J_omega_3_0 = J_omega[2]
    J_omega_4_0 = J_omega[3]
    J_omega_P_0 = J_omega[4]

    I = calculate_inertias()
    I_1_0 = I[0]
    I_2_0 = I[1]
    I_3_0 = I[2]
    I_4_0 = I[3]
    I_P_0 = I[4]

    M = simplify_and_substitute(m_1 * J_v_1_0.T * J_v_1_0
                                + m_2 * J_v_2_0.T * J_v_2_0
                                + m_3 * J_v_3_0.T * J_v_3_0
                                + m_4 * J_v_4_0.T * J_v_4_0
                                + m_p * J_v_P_0.T * J_v_P_0
                                + J_omega_1_0.T * I_1_0 * J_omega_1_0
                                + J_omega_2_0.T * I_2_0 * J_omega_2_0
                                + J_omega_3_0.T * I_3_0 * J_omega_3_0
                                + J_omega_4_0.T * I_4_0 * J_omega_4_0
                                + J_omega_P_0.T * I_P_0 * J_omega_P_0)

    C = sp.zeros(6)
    for k in range(C.shape[0]):
        for j in range(C.shape[0]):
            c_jk = sp.zeros(len(rho), 1)

            for i in range(len(c_jk)):
                c_ijk_expr = 1 / 2 * (sp.diff(M[k, j], rho[i]) + sp.diff(M[k, i], rho[j]) - sp.diff(M[i, j], rho[k]))
                # c_ijk = simplify_and_substitute(c_ijk_expr)
                c_jk[i] = c_ijk_expr

            C_kj_expr = (c_jk.T * rho_dot)[0]
            # C_kj = simplify_and_substitute((c_jk.T * rho_dot)[0])
            # C_kj = sp.nsimplify(C_kj)
            C_kj = C_kj_expr
            C[k, j] = C_kj

    C = sp.nsimplify(simplify_and_substitute(C))

    r_CoM = symbolic_forward_kinematics_positions('CoM')
    r_CoM_P_0 = r_CoM[4]
    g_vec = sp.Matrix([0, 0, -g_0])
    U = - m_p * r_CoM_P_0.T * g_vec
    g = U.jacobian(rho).T

    return M, C, g


def projection_matrix() -> Tuple[sp.Matrix, sp.Matrix]:
    t, _, _, _, rho, _, _, _ = define_symbols()

    q_1 = rho[0]
    beta_1 = rho[1]
    q_2 = rho[4]
    beta_2 = rho[5]

    den = sp.sin(q_1 - q_2 + beta_1 - beta_2)
    J_beta_11 = - sp.sin(q_1 - q_2 - beta_2) / den - 1
    J_beta_12 = - sp.sin(beta_2) / den
    J_beta_21 = sp.sin(beta_1) / den
    J_beta_22 = - sp.sin(q_1 - q_2 + beta_1) / den - 1
    R = sp.Matrix([[1,         0, 0, 0],
                   [J_beta_11, 0, 0, J_beta_12],
                   [0,         1, 0, 0],
                   [0,         0, 1, 0],
                   [0,         0, 0, 1],
                   [J_beta_21, 0, 0, J_beta_22]])
    R_dot = sp.simplify(sp.diff(R, t))

    return R, R_dot


def reduced_dynamic_matrices() -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    M, C, g = calculate_dynamical_matrices()
    R, R_dot = projection_matrix()

    M_bar = simplify_and_substitute(R.T * M * R)
    C_bar = simplify_and_substitute(R.T * C * R + R.T * M * R_dot)
    g_bar = simplify_and_substitute(R.T * g)

    return M_bar, C_bar, g_bar


# M, C, g = calculate_dynamical_matrices()
