import numpy as np
from scipy import linalg
import control as ct
import cvxpy as cp

from functools import partial
from typing import Callable, Dict, Optional, Tuple
from numpy.typing import NDArray

from physical_modelling.dynamics import continuous_inverse_dynamics, continuous_state_space_dynamics, discrete_forward_dynamics
from physical_modelling.linearisation import continuous_linear_state_space_representation, cont2discrete_zoh


def lqr(A, B, Q, R, q, q_dot, q_eq, q_dot_eq):
    x = np.concatenate([q, q_dot])
    x_eq = np.concatenate([q_eq, q_dot_eq])

    K, S, E = ct.dlqr(A, B, Q, R)
    tau = - K @ (x - x_eq)

    return tau


def lqr_input(t: float, x: NDArray, x_eq: NDArray, u_eq: NDArray, Q: NDArray, R: NDArray,
              params: Dict, continuous: bool = False, dt: float = 0.001) -> NDArray:
    """

    Computes the control input for the LQR controller.

    :param float t: time (passed in dynamics equation)
    :param NDArray x: state vector of shape (n,) (passed in dynamics equation)
    :param NDArray x_eq: equilibrium state vector of shape (n,) for set-point or (n, M) for trajectory tracking
    :param NDArray u_eq: equilibrium input vector of shape (m,) for set-point or (m, M) for trajectory tracking
    :param NDArray Q: state cost matrix of shape (n, n)
    :param NDArray R: input cost matrix of shape (m, m)
    :param Dict params: dictionary of physical parameters
    :param bool continuous: flag to use continuous or discrete LQR
                            (default: False, so if dt is passed, this argument can be omitted)
    :param float dt: sampling time for discrete LQR (can be omitted if continuous is True)

    :return: control input vector of shape (m,)
    :rtype: NDArray
    """
    A, B, C, D = continuous_linear_state_space_representation(params)
    Ad, Bd, _, _ = cont2discrete_zoh(dt, A, B, C, D)
    K_cont, _, _ = ct.lqr(A, B, Q, R)
    K_discrete, _, _ = ct.dlqr(Ad, Bd, Q, R)
    if continuous:
        K = K_cont
    else:
        K = K_discrete
        A, B, C, D = Ad, Bd, C, D

    u = - K @ (x - x_eq)
    # dx_dt = A @ (x - x_eq) + B @ u
    #
    # return dx_dt

    return u


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


def _generate_prediction_matrices(A: NDArray, B: NDArray, N: int) -> Tuple[NDArray, NDArray]:
    """

    Generates the prediction matrices for the optimal control problem for a constant reference.
    The state sequence x_N_plus_1 = [x(0), x(1), ..., x(N)]^T can be calculated from the initial state x(0) and the
    input sequence u_N = u(0), u(1), ..., u(N-1) as follows:

    x_N_plus_1 = T @ x(0) + S @ u_N

    :param NDArray A: state matrix of shape (n, n)
    :param NDArray B: input matrix of shape (n, m)
    :param int N: length of the receding horizon

    :return: prediction matrices T, S
        - T (:py:class:`NDArray`): prediction matrix from initial state of shape (n * (N + 1), n)
        - S (:py:class:`NDArray`): prediction matrix from input sequence of shape (n * (N + 1), m * N)
    :rtype: Tuple[NDArray, NDArray]
    """

    # Predicted state at time k with LTI system:
    # x(k) = A^k x(0) + Σ_{j=0}^{k-1} A^(k-j-1) B u(j)

    n, m = B.shape

    # Initialize matrices
    T = np.zeros((n * (N + 1), n))
    S = np.zeros((n * (N + 1), m * N))

    power_matrices = [np.eye(n)]
    for k in range(N + 1):
        # power_matrices[k] = A^k
        if k < N:
            power_matrices.append(power_matrices[k] @ A)

        # Prediction matrix from initial state (T)
        T[k * n:(k + 1) * n, :] = power_matrices[k]

        # Prediction matrix from input sequence (S)
        for j in range(k):
            S[k * n:(k + 1) * n, j * m:(j + 1) * m] = power_matrices[k - j - 1] @ B

    return T, S


def _generate_prediction_matrices_for_trajectory(A_list: NDArray, B_list: NDArray, N: int) -> Tuple[NDArray, NDArray]:
    """

    Generates the prediction matrices for the optimal control problem for a trajectory reference.
    The state sequence x_N_plus_1 = [x(0), x(1), ..., x(N)]^T can be calculated from the initial state x(0) and the
    input sequence u_N = u(0), u(1), ..., u(N-1) as follows:

    x_N_plus_1 = T @ x(0) + S @ u_N

    Because the system is time-varying, the prediction matrices T and S are time-varying as well,
    at each time step k the prediction starts from A(0) = A(k).

    :param NDArray A_list: list of state matrices of shape (n, n, M)
    :param NDArray B_list: list of input matrices of shape (n, m, M)
    :param int N: length of the receding horizon

    :return: prediction matrices T_list, S_list
        - T_list (:py:class:`NDArray`): prediction matrix from initial state of shape (n * (N + 1), n, M - N)
        - S_list (:py:class:`NDArray`): prediction matrix from input sequence of shape (n * (N + 1), m * N, M - N)
    :rtype: Tuple[NDArray, NDArray]
    """

    # Predicted state at time k with LTV system

    n, m, M = B_list.shape
    
    # Initialize matrices
    T_list = np.zeros((n * (N + 1), n, M - N))
    S_list = np.zeros((n * (N + 1), m * N, M - N))

    for t in range(M - N):
        # Prediction matrix from initial state (T_list)
        mem_T = np.eye(n)  # represents one n x n block
        T_list[0:n, :, t] = mem_T
        for k in range(N):
            T_list[(k + 1) * n:(k + 2) * n, :, t] = A_list[:, :, t + k] @ mem_T
            mem_T = A_list[:, :, t + k] @ mem_T

        # Prediction matrix from input sequence (S_list)
        mem_S = np.zeros((n, m * N))  # represents an n x mN row of n x m blocks
        for k in range(N):
            S_list[(k + 1) * n:(k + 2) * n, :, t] = A_list[:, :, t + k] @ mem_S
            S_list[(k + 1) * n:(k + 2) * n, k * m: (k + 1) * m, t] = B_list[:, :, t + k]
            mem_S = S_list[(k + 1) * n:(k + 2) * n, :, t]

    return T_list, S_list


def _generate_cost_matrices(x_0: NDArray, N: int,
                            T: NDArray, S: NDArray, Q: NDArray, R: NDArray, P: NDArray) -> Tuple[NDArray, NDArray]:
    """

    Generates the cost matrices for the optimal control problem.

    :param NDArray x_0: initial state (error) of shape (n,)
    :param int N: length of the receding horizon
    :param NDArray T: prediction matrix from initial state of shape (n * (N + 1), n)
    :param NDArray S: prediction matrix from input sequence of shape (n * (N + 1), m * N)
    :param NDArray Q: state cost matrix for stage cost of shape (n, n)
    :param NDArray R: input cost matrix for stage cost of shape (m, m)
    :param NDArray P: state cost matrix for terminal cost of shape (n, n)

    :return: cost matrices H, h (coefficients of the quadratic cost function)
        - H (:py:class:`NDArray`): matrix coefficient of the quadratic term of shape (m * N, m * N)
        - h (:py:class:`NDArray`): vector coefficient of the linear term of shape (m * N,)
    :rtype: Tuple[NDArray, NDArray]
    """
    n = x_0.shape[0]

    # block cost matrices corresponding to the state and input sequences x_N_plus_1 and u_N
    # (the last block in Q corresponds to the terminal cost)
    Q_bar = np.block([[np.kron(np.eye(N), Q), np.zeros((n * N, n))],
                      [np.zeros((n, n * N)),  P]])
    R_bar = np.kron(np.eye(N), R)

    # cost matrices for the optimal control problem
    H = S.T @ Q_bar @ S + R_bar
    h = S.T @ Q_bar @ T @ x_0

    # if trajectory_tracking:
    #     H = np.array([S[:, :, k].T @ Q_bar @ S[:, :, k] + R_bar for k in range(T.shape[2])])
    #     h = np.array([S[:, :, k].T @ Q_bar @ T[:, :, k] @ x_0 for k in range(T.shape[2])])
    # else:

    # c = 1 / 2 * x_0.T @ T.T @ Q_bar @ T @ x_0

    return H, h


class ConstraintConstants:
    def __init__(self, u_lim: NDArray, max_linearization_error: NDArray,
                 max_linearization_error_final: NDArray, max_error_q_dot_final: NDArray):
        self.u_lim = u_lim
        self.max_linearization_error = max_linearization_error
        self.max_linearization_error_final = max_linearization_error_final
        self.max_error_q_dot_final = max_error_q_dot_final

def _solve_optimal_control_problem(x_0: NDArray, N: int, x_ref_N_plus_1: NDArray,
                                   T: NDArray, S: NDArray, Q: NDArray, R: NDArray, P: NDArray,
                                   constraint_constants: ConstraintConstants) -> Tuple[NDArray, float]:
    """

    Solves the optimal control problem for a given horizon N and initial state x_0.

    :param NDArray x_0: initial state (error) of shape (n,)
    :param int N: length of the receding horizon
    :param NDArray x_ref_N_plus_1: reference state sequence until time N of shape (n, N + 1)
                                   or reference state of shape (n,)
    :param NDArray T: prediction matrix from initial state of shape (n * (N + 1), n)
    :param NDArray S: prediction matrix from input sequence of shape (n * (N + 1), m * N)
    :param NDArray Q: state cost matrix for stage cost of shape (n, n)
    :param NDArray R: input cost matrix for stage cost of shape (m, m)
    :param NDArray P: state cost matrix for terminal cost of shape (n, n)
    :param ConstraintConstants constraint_constants: constraint constants

    :return: solution of the optimal control problem
        - u_N_opt (:py:class:`NDArray`): optimal control input sequence of shape (m * N,)
        - V_N_opt (:py:class:`float`): optimal cost value
    :rtype: (NDArray, float)
    """
    n = x_0.shape[0]
    m = R.shape[0]

    if x_ref_N_plus_1.ndim == 1:
        x_ref_N_plus_1 = np.tile(x_ref_N_plus_1[:, np.newaxis], N + 1)

    # ===========================
    # Optimization variable: u_N (control input sequence over the horizon)
    u_N = cp.Variable(m * N)

    # ===========================
    # Cost function
    H, h = _generate_cost_matrices(x_0, N, T, S, Q, R, P)
    cost_fn = 1/2 * cp.quad_form(u_N, H) + h.T @ u_N

    # ===========================
    # Input constraints
    u_lim = constraint_constants.u_lim
    u_lim_vec = np.tile(u_lim, N)
    input_constraints = [ u_N <= u_lim_vec,
                         -u_N <= u_lim_vec]

    # ===========================
    # State constraints
    #   for one state: W @ x(k) <= d(k)
    #   for N states: W_bar @ x_N <= d_N

    # state selection matrix
    W = np.array([[ 0,  1,  0,  0, 0, 0, 0, 0],
                  [ 0,  0,  1,  0, 0, 0, 0, 0],
                  [ 0, -1,  0,  0, 0, 0, 0, 0],
                  [ 0,  0, -1,  0, 0, 0, 0, 0],
                  [ 1,  0,  0, -1, 0, 0, 0, 0],
                  [-1,  0,  0,  1, 0, 0, 0, 0]])
    W_bar = np.kron(np.eye(N), W)

    # constraint vector sequence for the state sequence
    max_linearization_error = constraint_constants.max_linearization_error
    d_N = np.hstack([np.hstack([max_linearization_error + x_ref_N_plus_1[1:3, k],
                                max_linearization_error - x_ref_N_plus_1[1:3, k],
                                np.pi,
                                np.pi]) for k in range(N)])

    # T_N: first N blocks of T corresponding to the state sequence x_N = [x(0), x(1), ..., x(N-1)]
    # S_N: first N block rows of S corresponding to the state sequence x_N = [x(0), x(1), ..., x(N-1)]
    T_N = T[:n * N, :]
    S_N = S[:n * N, :]

    state_constraints = [W_bar @ S_N @ u_N <= - W_bar @ T_N @ x_0 + d_N]

    # ===========================
    # Terminal constraints
    #   W_terminal @ x(N) <= d(N)

    # state selection matrix
    W_terminal = np.block([[ 0,  1,  0,  0,  0,  0,  0,  0],
                           [ 0,  0,  1,  0,  0,  0,  0,  0],
                           [ 0, -1,  0,  0,  0,  0,  0,  0],
                           [ 0,  0, -1,  0,  0,  0,  0,  0],
                           [np.zeros((4, 4)),  np.eye(4)],
                           [np.zeros((4, 4)), -np.eye(4)],
                           [ 1,  0,  0, -1,  0,  0,  0,  0],
                           [-1,  0,  0,  1,  0,  0,  0,  0]])
    # constraint vector for the terminal state
    max_linearization_error_final = constraint_constants.max_linearization_error_final
    max_error_q_dot_final = constraint_constants.max_error_q_dot_final
    d_terminal = np.hstack([max_linearization_error_final + x_ref_N_plus_1[1:3, N],
                            max_linearization_error_final - x_ref_N_plus_1[1:3, N],
                            max_error_q_dot_final + x_ref_N_plus_1[4:, N],
                            max_error_q_dot_final - x_ref_N_plus_1[4:, N],
                            np.pi,
                            np.pi])

    # T_final: last block of T corresponding to the state x(N)
    # S_final: last block row of S corresponding to the state x(N)
    T_final = T[n * N:, :]
    S_final = S[n * N:, :]

    terminal_constraints = [W_terminal @ S_final @ u_N <= - W_terminal @ T_final @ x_0 + d_terminal]

    # ===========================
    # Solve the optimization problem
    constraints = input_constraints + state_constraints + terminal_constraints
    # constraints = []
    prob = cp.Problem(cp.Minimize(cost_fn), constraints)
    try:
        V_N_opt = prob.solve(solver='ECOS', verbose=False)
        # V_N_opt = prob.solve(verbose=False)
    except cp.error.SolverError:
        print("Solution failed")
        V_N_opt = prob.solve(solver='ECOS', verbose=True)
        # V_N_opt = prob.solve(verbose=True)

    # V_N_opt = prob.solve(solver='ECOS', verbose=True)
    u_N_opt = u_N.value

    return u_N_opt, V_N_opt


def mpc(t_ts: NDArray, y_ref: NDArray, u_ref: NDArray, x_ic: NDArray, N: int, dt: float,
        Q: NDArray, R: NDArray, P: NDArray | None, constraint_constants: ConstraintConstants,
        params: Dict[str, float], return_states: bool = False) -> Tuple[NDArray, ...]:
    """

    Implements the receding horizon MPC algorithm with the following steps:

    1. measure current state
    2. solve optimal control problem
    3. apply MPC control law
    4. update dynamical system
    5. repeat for all time steps

    :param NDArray t_ts: vector of time steps of shape (M,)
    :param NDArray y_ref: reference output vector of shape (n,) or reference output trajectory of shape (n, M)
    :param NDArray u_ref: reference input vector of shape (m,) or reference input trajectory of shape (m, M)
    :param NDArray x_ic: initial condition of the state vector of shape (n,)
    :param int N: length of the receding horizon
    :param float dt: sampling time
    :param NDArray Q: state cost matrix for stage cost of shape (n, n)
    :param NDArray R: input cost matrix for stage cost of shape (m, m)
    :param NDArray or None P: state cost matrix for terminal cost of shape (n, n) (if None, DARE is solved for P)
    :param ConstraintConstants constraint_constants: constraint constants
    :param Dict[str, float] params: dictionary of physical parameters
    :param bool return_states: boolean flag to return the states

    :returns:
        - t_ts (:py:class:`NDArray`): time steps of shape (N - M + 1,)
        - y_all (:py:class:`NDArray`): states of shape (n, M - N + 1)
        - u_all (:py:class:`NDArray`): inputs of shape (m, M - N + 1)
        - x_all (:py:class:`NDArray`): states of shape (n, M - N + 1) in case return_states is True
    :rtype: Tuple[NDArray, ...]
    """
    M = t_ts.shape[0]
    n = x_ic.shape[0]
    m = u_ref.shape[0]

    # ===========================
    # Initialize state, output, and input error vectors
    x_e_all = np.zeros((n, M - N + 1))
    y_e_all = np.zeros((n, M - N + 1))
    u_e_all = np.zeros((m, M - N + 1))
    x_all = np.zeros((n, M - N + 1))
    y_all = np.zeros((n, M - N + 1))
    u_all = np.zeros((m, M - N + 1))

    # ===========================
    # Trajectory tracking or set-point tracking
    if y_ref.ndim == 1:
        trajectory_tracking = False
        y_ref = np.tile(y_ref, (M, 1)).T
        u_ref = np.tile(u_ref, (M, 1)).T
    else:
        trajectory_tracking = True
        M_y = y_ref.shape[1]
        M_u = u_ref.shape[1]
        assert M_y == M, "Reference output trajectory length must be equal to the simulation length"
        assert M_u == M, "Reference input trajectory length must be equal to the simulation length"

    # ===========================
    # All states are measured
    x_ref = y_ref

    # ===========================
    # State space representation (linearisation) at each time step
    # Ad, Bd, Cd, Dd = zip(*(cont2discrete_zoh(dt, *continuous_linear_state_space_representation(
    #     params, x_eq=x_ref[:, t_idx], tau_eq=u_ref[:, t_idx], trajectory_tracking=trajectory_tracking))
    #                        for t_idx in range(M)))
    Ad, Bd, Cd, Dd = zip(*(cont2discrete_zoh(dt, *continuous_linear_state_space_representation(params))
                           for _ in range(M)))

    # ===========================
    # Set initial errors
    x_e_all[:, 0] = x_ic - x_ref[:, 0]
    y_e_all[:, 0] = Cd[0] @ x_ic - y_ref[:, 0]
    x_all[:, 0] = x_ic
    y_all[:, 0] = Cd[0] @ x_ic

    # ===========================
    # Generate prediction matrices and cost matrix for the terminal cost
    if trajectory_tracking:
        T, S = _generate_prediction_matrices_for_trajectory(Ad, Bd, N)
        assert P is not None, "Terminal cost matrix P must be provided for trajectory tracking"
    else:
        T, S = _generate_prediction_matrices(Ad[0], Bd[0], N)
        if P is None:
            # P = ct.dare(Ad[0], Bd[0], Q, R)
            P = linalg.solve_discrete_are(Ad[0], Bd[0], Q, R)

    # ===========================
    # Solve the optimal control problem and apply the MPC control law for all time steps
    for t_idx in range(M - N):
        # --------------------------
        # 1. measure current state
        # x_0 = y_all[:, t_idx]
        x_0 = y_e_all[:, t_idx]
        x_0 = x_all[:, t_idx] - x_ref[:, t_idx]

        # --------------------------
        # 2. solve optimal control problem
        # reference state sequence for the horizon (required for state constraints)
        x_ref_t_N_plus_1 = x_ref[:, t_idx: t_idx + N + 1]

        # select current prediction matrices
        if trajectory_tracking:
            T_t = T[:, :, t_idx]
            S_t = S[:, :, t_idx]
            x_ref_t = x_ref_t_N_plus_1
        else:
            T_t = T
            S_t = S
            # x_ref_t = x_ref[:, t_idx]
            x_ref_t = x_ref_t_N_plus_1

        # solve the optimal control problem
        u_N_opt, V_N_opt = _solve_optimal_control_problem(x_0, N, x_ref_t, T_t, S_t, Q, R, P, constraint_constants)

        # --------------------------
        # 3. MPC control law
        u_opt = u_N_opt[:m]

        # --------------------------
        # 4. update dynamical system
        # with linear dynamics
        x_next = Ad[t_idx] @ x_all[:, t_idx] + Bd[t_idx] @ u_opt
        y_next = Cd[t_idx] @ x_all[:, t_idx] + Dd[t_idx] @ u_opt

        # # with nonlinear dynamics
        # # q_e_next, q_e_dot_next, _ = discrete_forward_dynamics(params, dt, x_0[:4], x_0[4:], u_opt)
        # # x_e_next = np.concatenate([q_e_next, q_e_dot_next])
        # # y_e_next = x_e_next

        # --------------------------
        # save results
        u_all[:, t_idx] = u_opt
        x_all[:, t_idx + 1] = x_next
        y_all[:, t_idx + 1] = y_next


    if return_states:
        return t_ts[:M - N + 1], y_all, u_all, x_all
    else:
        return t_ts[:M - N + 1], y_all, u_all


def discrete_input_for_continuous_nonlinear_simulation(t: float, y: NDArray, y_ref: NDArray, u_ref: NDArray, N: int,
                                                       M: int, dt: float, Q: NDArray, R: NDArray, P: NDArray | None,
                                                       constraint_constants: ConstraintConstants,
                                                       params: Dict[str, float]) -> NDArray:
    """

    Calculate the discrete control input for the continuous nonlinear system simulation.

    :param float t: current time
    :param NDArray y: current output of shape (n,)
    :param NDArray y_ref: reference output vector of shape (n,) or reference output trajectory of shape (n, M)
    :param NDArray u_ref: reference input vector of shape (m,) or reference input trajectory of shape (m, M)
    :param int N: length of the receding horizon
    :param int M: number of time steps (length of the reference trajectory)
    :param float dt: sampling time
    :param NDArray Q: state cost matrix for stage cost of shape (n, n)
    :param NDArray R: input cost matrix for stage cost of shape (m, m)
    :param NDArray or None P: state cost matrix for terminal cost of shape (n, n) (if None, DARE is solved for P)
    :param ConstraintConstants constraint_constants: constraint constants
    :param Dict[str, float] params: dictionary of physical parameters

    :return: control input of shape (m,)
    :rtype: NDArray
    """
    m = u_ref.shape[0]

    # ===========================
    # Initialize static variables k and u_k to change input only at discrete times
    if not hasattr(discrete_input_for_continuous_nonlinear_simulation, "k"):
        discrete_input_for_continuous_nonlinear_simulation.k = 0
    if not hasattr(discrete_input_for_continuous_nonlinear_simulation, "u_k"):
        discrete_input_for_continuous_nonlinear_simulation.u_k = np.zeros(m)

    # ===========================
    # Trajectory tracking or set-point tracking
    if y_ref.ndim == 1:
        trajectory_tracking = False
        y_ref = np.tile(y_ref, (M, 1)).T
        u_ref = np.tile(u_ref, (M, 1)).T
    else:
        trajectory_tracking = True
        M_y = y_ref.shape[1]
        M_u = u_ref.shape[1]
        assert M == M_y, "Reference output trajectory must have length M"
        assert M == M_u, "Reference input trajectory must have length M"

    # ===========================
    # Solve the optimal control problem only at discrete times
    k = discrete_input_for_continuous_nonlinear_simulation.k  # to shorten the code
    if k * dt <= t:
        # --------------------------
        # 0. discrete state space representation
        x_ref = y_ref

        # Ad, Bd, Cd, Dd = zip(*(cont2discrete_zoh(dt, *continuous_linear_state_space_representation(
        #     params, x_eq=x_ref[:, t_idx], tau_eq=u_ref[:, t_idx], trajectory_tracking=trajectory_tracking))
        #                        for t_idx in range(k, k + N)))
        Ad, Bd, Cd, Dd = zip(*(cont2discrete_zoh(dt, *continuous_linear_state_space_representation(params))
                               for _ in range(N)))

        if trajectory_tracking:
            T, S = _generate_prediction_matrices_for_trajectory(Ad, Bd, N)
            assert T.shape[2] == 1, "Wrong range resulted in invalid matrix shape"
            T, S = T.squeeze(), S.squeeze()
            assert P is not None, "Terminal cost matrix P must be provided for trajectory tracking"
        else:
            Ad, Bd = Ad[0], Bd[0]
            T, S = _generate_prediction_matrices(Ad, Bd, N)
            if P is None:
                P = linalg.solve_discrete_are(Ad, Bd, Q, R)

        # --------------------------
        # 1. measure current state
        y_ref_0 = y_ref[:, k]
        x_ref_0 = y_ref_0
        x_0 = y
        x_e_0 = x_0 - x_ref_0

        # --------------------------
        # 2. solve optimal control problem
        # reference state (sequence) for the horizon (required for state constraints)
        x_ref_t = x_ref[:, k:k + N + 1] if trajectory_tracking else x_ref[:, k]

        # solve the optimal control problem
        u_N_e_opt, V_N_opt = _solve_optimal_control_problem(x_e_0, N, x_ref_t, T, S, Q, R, P, constraint_constants)
        if u_N_e_opt is None:
            print(f"Optimal control problem failed at time {t:.5f} (k = {k}/{M}), using previous control input")
            discrete_input_for_continuous_nonlinear_simulation.k += 1
            return discrete_input_for_continuous_nonlinear_simulation.u_k
        # --------------------------
        # 3. MPC control law
        u_e_0 = u_N_e_opt[:m]
        u_0 = u_e_0 + u_ref[:, k]

        # --------------------------
        # 4. update dynamical system
        # this is done by the ODE solver

        # --------------------------
        # update counter and control input
        discrete_input_for_continuous_nonlinear_simulation.k += 1
        discrete_input_for_continuous_nonlinear_simulation.u_k = u_0

    # return control input
    return discrete_input_for_continuous_nonlinear_simulation.u_k






