##

import numpy as np
import sympy as sp
import sympy.simplify.fu

from typing import List
from time import time


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

    assert axis in ['x', 'y', 'z', 'X', 'Y', 'Z'], "Invalid axis"

    if axis in ['x', 'X']:
        return _R_x

    if axis in ['y', 'Y']:
        return _R_y

    if axis in ['z', 'Z']:
        return _R_z


def simplify_and_substitute(expression, old: List, new: List):
    assert len(old) == len(new), "Number of old and new variables must match"

    subs_dict = dict(zip(old, new))

    # subbed = expression.subs(subs_dict)
    # subbed_and_simplified = sp.simplify(subbed)

    simplified = sp.simplify(expression)
    simplified_and_subbed = simplified.subs(subs_dict)

    result = simplified_and_subbed

    return result


t, g_0 = sp.symbols('t g_0')
I_1, I_2, I_3, I_4, I_px, I_pz, m_1, m_2, m_3, m_4, m_p, l_c1, l_c2, l_c3, l_c4, l_cp, L \
    = sp.symbols('I_1, I_2, I_3, I_4, I_px, I_pz, m_1, m_2, m_3, m_4, m_p, l_c1, l_c2, l_c3, l_c4, l_cp, L')
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

substitution_old = [q_1_dot, phi_dot, theta_dot, q_2_dot, q_1_ddot, phi_ddot, theta_ddot, q_2_ddot,
                    beta_1_dot, beta_2_dot, beta_1_ddot, beta_2_ddot]
substitution_new = [q_1_dot_, phi_dot_, theta_dot_, q_2_dot_, q_1_ddot_, phi_ddot_, theta_ddot_, q_2_ddot_,
                    beta_1_dot_, beta_2_dot_, beta_1_ddot_, beta_2_ddot_]

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

##
t_ = time()

r_P_b_b = L * sp.Matrix([sp.sin(theta(t)),
                        -sp.cos(theta(t)) * sp.sin(phi(t)),
                         sp.cos(theta(t)) * sp.cos(phi(t))])
r_CoM_b_b = l_cp * sp.Matrix([sp.sin(theta(t)),
                             -sp.cos(theta(t)) * sp.sin(phi(t)),
                              sp.cos(theta(t)) * sp.cos(phi(t))])
R_b_0 = rotation_matrix('z', q_1(t) + beta_1(t))

# r_P_b_0 = R_b_0 * r_P_b_b

r_A_0 = sp.zeros(3, 1)
r_B_0 = sp.Matrix([L * sp.cos(q_1(t)),
                   L * sp.sin(q_1(t)),
                   0])
r_C_0 = sp.Matrix([L * sp.cos(q_1(t)) + L * sp.cos(q_1(t) + beta_1(t)),
                   L * sp.sin(q_1(t)) + L * sp.sin(q_1(t) + beta_1(t)),
                   0])
r_D_0 = sp.Matrix([2 * L + L * sp.cos(q_2(t)),
                   L * sp.sin(q_2(t)),
                   0])
r_E_0 = sp.Matrix([2 * L,
                   0,
                   0])
r_P_0 = r_C_0 + R_b_0 * r_P_b_b

r_CoM_1_0 = sp.Matrix([l_c1 * sp.cos(q_1(t)),
                       l_c1 * sp.sin(q_1(t)),
                       0])
r_CoM_2_0 = sp.Matrix([L * sp.cos(q_1(t)) + l_c2 * sp.cos(q_1(t) + beta_1(t)),
                       L * sp.sin(q_1(t)) + l_c2 * sp.sin(q_1(t) + beta_1(t)),
                       0])
r_CoM_3_0 = sp.Matrix([2*L + L * sp.cos(q_2(t)) + l_c3 * sp.cos(q_2(t) + beta_2(t)),
                       L * sp.sin(q_2(t)) + l_c3 * sp.sin(q_2(t) + beta_2(t)),
                       0])
r_CoM_4_0 = sp.Matrix([2*L + l_c4 * sp.cos(q_2(t)),
                       l_c4 * sp.sin(q_2(t)),
                       0])
r_CoM_P_0 = r_C_0 + R_b_0 * r_CoM_b_b

r_A_dot_0 = simplify_and_substitute(sp.diff(r_A_0, t), substitution_old, substitution_new)
r_B_dot_0 = simplify_and_substitute(sp.diff(r_B_0, t), substitution_old, substitution_new)
r_C_dot_0 = simplify_and_substitute(sp.diff(r_C_0, t), substitution_old, substitution_new)
r_D_dot_0 = simplify_and_substitute(sp.diff(r_D_0, t), substitution_old, substitution_new)
r_E_dot_0 = simplify_and_substitute(sp.diff(r_E_0, t), substitution_old, substitution_new)
r_P_dot_0 = simplify_and_substitute(sp.diff(r_P_0, t), substitution_old, substitution_new)

r_A_ddot_0 = simplify_and_substitute(sp.diff(r_A_0, t, 2), substitution_old, substitution_new)
r_B_ddot_0 = simplify_and_substitute(sp.diff(r_B_0, t, 2), substitution_old, substitution_new)
r_C_ddot_0 = simplify_and_substitute(sp.diff(r_C_0, t, 2), substitution_old, substitution_new)
r_D_ddot_0 = simplify_and_substitute(sp.diff(r_D_0, t, 2), substitution_old, substitution_new)
r_E_ddot_0 = simplify_and_substitute(sp.diff(r_E_0, t, 2), substitution_old, substitution_new)
r_P_ddot_0 = simplify_and_substitute(sp.diff(r_P_0, t, 2), substitution_old, substitution_new)

omega_1_0 = sp.Matrix([0, 0, q_1_dot])
omega_2_0 = sp.Matrix([0, 0, q_1_dot + beta_1_dot])
omega_3_0 = sp.Matrix([0, 0, q_2_dot + beta_2_dot])
omega_4_0 = sp.Matrix([0, 0, q_2_dot])

R_x = rotation_matrix('x', phi(t))
R_y = rotation_matrix('y', theta(t))
R_p_0 = R_b_0 * R_x * R_y
omega_P_0_cross = sp.simplify(sp.diff(R_p_0, t) * R_p_0.T)
omega_P_0 = sp.Matrix([omega_P_0_cross[2,1], omega_P_0_cross[0,2], omega_P_0_cross[1,0]])
omega_P_0_cross_simp = simplify_and_substitute(sp.diff(R_p_0, t) * R_p_0.T, substitution_old, substitution_new)
omega_P_0_simp = sp.Matrix([omega_P_0_cross_simp[2,1], omega_P_0_cross_simp[0,2], omega_P_0_cross_simp[1,0]])

t_1 = time() - t_
print(t_1)

##
t_ = time()

R_1_0 = rotation_matrix('z', q_1(t))
R_2_0 = rotation_matrix('z', q_1(t) + beta_1(t))
R_3_0 = rotation_matrix('z', q_2(t) + beta_2(t))
R_4_0 = rotation_matrix('z', q_2(t))

I_1_0 = R_1_0 * sp.diag(0, 0, I_1)
I_2_0 = R_1_0 * sp.diag(0, 0, I_2)
I_3_0 = R_1_0 * sp.diag(0, 0, I_3)
I_4_0 = R_1_0 * sp.diag(0, 0, I_4)
I_P_0 = R_p_0 * sp.diag(I_px, I_px, I_pz) * R_p_0.T

J_v_1_0 = r_CoM_1_0.jacobian(rho)
J_v_2_0 = r_CoM_2_0.jacobian(rho)
J_v_3_0 = r_CoM_3_0.jacobian(rho)
J_v_4_0 = r_CoM_4_0.jacobian(rho)
J_v_P_0 = r_CoM_P_0.jacobian(rho)
J_omega_1_0 = omega_1_0.jacobian(rho_dot)
J_omega_2_0 = omega_2_0.jacobian(rho_dot)
J_omega_3_0 = omega_3_0.jacobian(rho_dot)
J_omega_4_0 = omega_4_0.jacobian(rho_dot)
J_omega_P_0 = omega_P_0.jacobian(rho_dot)

t_2 = time() - t_
print(t_2)

##

t_ = time()

M = simplify_and_substitute(m_1 * J_v_1_0.T * J_v_1_0
                            + m_2 * J_v_2_0.T * J_v_2_0
                            + m_3 * J_v_3_0.T * J_v_3_0
                            + m_4 * J_v_4_0.T * J_v_4_0
                            + m_p * J_v_P_0.T * J_v_P_0
                            + J_omega_1_0.T * I_1_0 * J_omega_1_0
                            + J_omega_2_0.T * I_2_0 * J_omega_2_0
                            + J_omega_3_0.T * I_3_0 * J_omega_3_0
                            + J_omega_4_0.T * I_4_0 * J_omega_4_0
                            + J_omega_P_0.T * I_P_0 * J_omega_P_0, substitution_old, substitution_new)
t_3 = time() - t_
print(t_3)

t_ = time()

C = sp.zeros(6)
for k in range(C.shape[0]):
    for j in range(C.shape[0]):
        c_jk = sp.zeros(len(rho), 1)

        for i in range(len(c_jk)):
            c_ijk_expr = 1/2 * (sp.diff(M[k,j], rho[i]) + sp.diff(M[k,i], rho[j]) - sp.diff(M[i,j], rho[k]))
            # c_ijk = simplify_and_substitute(c_ijk_expr, substitution_old, substitution_new)
            c_jk[i] = c_ijk_expr

        # C_kj = simplify_and_substitute((c_jk.T * rho_dot)[0], substitution_old, substitution_new)
        # C_kj = sp.nsimplify(C_kj)
        C_kj = (c_jk.T * rho_dot)[0]
        C[k, j] = C_kj

C = sp.nsimplify(simplify_and_substitute(C, substitution_old, substitution_new))

t_4 = time() - t_
print(t_4)

g_vec = sp.Matrix([0, 0, -g_0])
U = - m_p * r_CoM_P_0.T * g_vec
g = U.jacobian(rho).T

pass


den = sp.sin(q_1(t) - q_2(t) + beta_1(t) - beta_2(t))
J_beta_11 = - sp.sin(q_1(t) - q_2(t) - beta_2(t)) / den - 1
J_beta_12 = - sp.sin(beta_2(t)) / den
J_beta_21 = sp.sin(beta_2(t)) / den
J_beta_22 = - sp.sin(q_1(t) - q_2(t) + beta_1(t)) / den - 1
R = sp.Matrix([[1, 0, 0, 0],
               [J_beta_11, 0, 0, J_beta_12],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1],
               [J_beta_21, 0, 0, J_beta_22]])
R_dot = sp.simplify(sp.diff(R, t))

t_ = time()
M_bar = simplify_and_substitute(R.T * M * R, substitution_old, substitution_new)
t_5 = time() - t_
print(t_5)

t_ = time()
C_bar = simplify_and_substitute(R.T * C * R + R.T * M * R_dot, substitution_old, substitution_new)
t_6 = time() - t_
print(t_6)

t_ = time()
g_bar = simplify_and_substitute(R.T * g, substitution_old, substitution_new)
t_7 = time() - t_
print(t_7)

pass


##

