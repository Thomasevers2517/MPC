import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from controllers import mpc
import physical_modelling.initialisation as init

LINE_WIDTH = 1

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times"
})


def plot_effect_of_mpc_horizon(t_ts, y_ref, u_ref, x_0, dt, Q, R, P, constraints, physical_parameters, N_list, filepath=None):
    y_MPC_list = []
    t_MPC_list = []
    for N in N_list:
        t_MPC, y_MPC, u_MPC = mpc(t_ts, y_ref, u_ref, x_0, N, dt, Q, R, P, constraints, physical_parameters)
        y_MPC_list.append(y_MPC)
        t_MPC_list.append(t_MPC)

    q_1_ref = y_ref[0]
    phi_ref = y_ref[1]
    q_1_list = [y[0, :] for y in y_MPC_list]
    phi_list = [y[1, :] for y in y_MPC_list]

    plt.title("Effect of MPC horizon on the tracking of the first link angle")

    for i, phi in enumerate(phi_list):
        # plt.step(t_MPC_list[i], phi - phi_ref, label=rf"$N = {N_list[i]}$")
        plt.step(t_MPC_list[i], q_1_list[i] - q_1_ref, label=rf"$N = {N_list[i]}$")

    plt.xlabel("t [s]")
    # plt.ylabel(r"$\tilde{\phi} ~\mathrm{[rad]}$")
    plt.ylabel(r"$\tilde{q}_1 ~\mathrm{[rad]}$")
    plt.legend()
    plt.grid()

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')

    plt.show()

def plot_effect_of_mpc_weighting_matrices(t_ts, y_ref, u_ref, x_0, dt, N, Q, R, P, constraints, physical_parameters,
                                          which_matrix, matrix_values, filepath=None):
    y_MPC_list = []
    for matrix_value in matrix_values:
        if which_matrix == "Q":
            Q = matrix_value * np.eye(8)
        elif which_matrix == "R":
            R = matrix_value * np.eye(2)
        elif which_matrix == "P":
            P = matrix_value * np.eye(8)
        t_MPC, y_MPC, u_MPC = mpc(t_ts, y_ref, u_ref, x_0, N, dt, Q, R, P, constraints, physical_parameters)
        y_MPC_list.append(y_MPC)

    q_1_ref = y_ref[0]
    phi_ref = y_ref[1]

    q_1_list = [y[0, :] for y in y_MPC_list]
    phi_list = [y[1, :] for y in y_MPC_list]

    fig, axes = plt.subplots(2, 1, sharex=True)

    plt.title(r"Effect of cost matrix $" + f"{which_matrix} = {which_matrix.lower()}" +
              r"I$ on the tracking of the generalised coordinates")

    for i, (q_1, phi) in enumerate(zip(q_1_list, phi_list)):
        axes[0].step(t_MPC, q_1 - q_1_ref, label=rf"${which_matrix.lower()} = {matrix_values[i]}$")
        axes[1].step(t_MPC, phi - phi_ref, label=rf"${which_matrix.lower()} = {matrix_values[i]}$")

    axes[0].set_ylabel(r"$\tilde{q}_1 ~\mathrm{[rad]}$")
    axes[1].set_ylabel(r"$\tilde{\phi} ~\mathrm{[rad]}$")
    axes[-1].set_xlabel("t [s]")
    axes[-1].legend()
    axes[0].grid()
    axes[1].grid()
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')

    plt.show()

def plot_effect_of_R_on_u(t_ts, y_ref, u_ref, x_0, dt, N, Q, P, constraints, physical_parameters, R_values, filepath=None):
    y_MPC_list = []
    u_MPC_list = []
    for R_value in R_values:
        R = R_value * np.eye(2)
        t_MPC, y_MPC, u_MPC = mpc(t_ts, y_ref, u_ref, x_0, N, dt, Q, R, P, constraints, physical_parameters)
        y_MPC_list.append(y_MPC)
        u_MPC_list.append(u_MPC)

    u_1_ref = u_ref[0]
    u_1_list = [u[0, :] for u in u_MPC_list]

    plt.title(r"Effect of cost matrix $R = r I$ on the control input")

    for i, u in enumerate(u_1_list):
        plt.step(t_MPC, u - u_1_ref, label=rf"$r = {R_values[i]}$")

    plt.xlabel("t [s]")
    plt.ylabel(r"$\tilde{\tau_1} ~\mathrm{[Nm]}$")
    plt.legend()
    plt.grid()

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')

    plt.show()


def plot_effect_of_dt(t_ts, y_ref, u_ref, x_0, dt_values, N, Q, R, P, constraints, physical_parameters, filepath=None):
    t_MPC_list = []
    y_MPC_list = []
    for dt in dt_values:
        t_ts_i = init.generate_t_ts(dt, t_ts[-1])[0]
        t_MPC, y_MPC, u_MPC = mpc(t_ts_i, y_ref, u_ref, x_0, N, dt, Q, R, P, constraints, physical_parameters)
        t_MPC_list.append(t_MPC)
        y_MPC_list.append(y_MPC)

    q_1_ref = y_ref[0]
    phi_ref = y_ref[1]
    q_1_list = [y[0, :] for y in y_MPC_list]
    phi_list = [y[1, :] for y in y_MPC_list]

    fig, axes = plt.subplots(2, 1, sharex=True)

    plt.title("Effect of MPC sampling time on the tracking of the generalised coordinates")

    for i, (q_1, phi) in enumerate(zip(q_1_list, phi_list)):
        axes[0].step(t_MPC_list[i], q_1 - q_1_ref, label=rf"$T_s = {dt_values[i]}" + r"~\mathrm{s}$")
        axes[1].step(t_MPC_list[i], phi - phi_ref, label=rf"$T_s = {dt_values[i]}" + r"~\mathrm{s}$")

    axes[0].set_ylabel(r"$\tilde{q}_1 ~\mathrm{[rad]}$")
    axes[1].set_ylabel(r"$\tilde{\phi} ~\mathrm{[rad]}$")
    axes[-1].set_xlabel("t [s]")
    axes[-1].legend()
    axes[0].grid()
    axes[1].grid()
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')

    plt.show()


def plot_error_mpc_vs_lqr(t: NDArray, y_mpc: NDArray, y_lqr: NDArray, q_des: NDArray, filepath: str = None) -> None:
    q_1_mpc = y_mpc[0, :]
    phi_mpc = y_mpc[1, :]
    theta_mpc = y_mpc[2, :]
    q_2_mpc = y_mpc[3, :]

    q_1_lqr = y_lqr[0, :]
    phi_lqr = y_lqr[1, :]
    theta_lqr = y_lqr[2, :]
    q_2_lqr = y_lqr[3, :]

    q_1_des = q_des[0]
    phi_des = q_des[1]
    theta_des = q_des[2]
    q_2_des = q_des[3]

    fig, axes = plt.subplots(4, 1, num="Comparison of MPC and LQR", sharex=True) #figsize=(10, 5)

    plt.suptitle("Comparison of MPC and LQR controlled system")

    axes[0].step(t, q_1_mpc - q_1_des, color='b', linewidth=LINE_WIDTH, label=r"$q_1^\mathrm{MPC}$")
    axes[0].step(t, q_1_lqr - q_1_des, color='r', linewidth=LINE_WIDTH, label=r"$q_1^\mathrm{LQR}$")
    # axes[0].step(t, q_1_des, color='g', linestyle='--', linewidth=LINE_WIDTH, label=r"$q_1^\mathrm{d}$")
    # axes[0].legend()

    axes[1].step(t, q_2_mpc - q_2_des, color='b', linewidth=LINE_WIDTH, label=r"$q_2^\mathrm{MPC}$")
    axes[1].step(t, q_2_lqr - q_2_des, color='r', linewidth=LINE_WIDTH, label=r"$q_2^\mathrm{LQR}$")
    # axes[1].step(t, q_2_des, color='g', linestyle='--', linewidth=LINE_WIDTH, label=r"$q_2^\mathrm{d}$")
    # axes[1].legend()

    axes[2].step(t, phi_mpc - phi_des, color='b', linewidth=LINE_WIDTH, label=r"$\phi^\mathrm{MPC}$")
    axes[2].step(t, phi_lqr - phi_des, color='r', linewidth=LINE_WIDTH, label=r"$\phi^\mathrm{LQR}$")
    # axes[2].step(t, phi_des, color='g', linestyle='--', linewidth=LINE_WIDTH, label=r"$\phi^\mathrm{d}$")
    # axes[2].legend()

    axes[3].step(t, theta_mpc - theta_des, color='b', linewidth=LINE_WIDTH, label=r"$\theta^\mathrm{MPC}$")
    axes[3].step(t, theta_lqr - theta_des, color='r', linewidth=LINE_WIDTH, label=r"$\theta^\mathrm{LQR}$")
    # axes[3].step(t, theta_des, color='g', linestyle='--', linewidth=LINE_WIDTH, label=r"$\theta^\mathrm{d}$")
    axes[3].legend()

    axes[-1].set_xlabel("t [s]")
    axes[0].set_ylabel(r"$\tilde{q}_1 ~\mathrm{[rad]}$")
    axes[1].set_ylabel(r"$\tilde{q}_2 ~\mathrm{[rad]}$")
    axes[2].set_ylabel(r"$\tilde{\phi} ~\mathrm{[rad]}$")
    axes[3].set_ylabel(r"$\tilde{\theta} ~\mathrm{[rad]}$")

    for ax in axes:
        ax.set_xlim(t[0], t[-1])
        ax.grid()

    plt.tight_layout()
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif"
    # })

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')

    plt.show()


def plot_errors(
    t: NDArray,  # 1xN array with time instances
    q: NDArray,  # 4xN array with generalised coordinates at each time instance
    q_des: NDArray,  # 4x1 array with desired generalised coordinates
    filepath: str = None,
) -> None:
    q_1 = q[0, :]
    phi = q[1, :]
    theta = q[2, :]
    q_2 = q[3, :]

    q_1_des = q_des[0]
    phi_des = q_des[1]
    theta_des = q_des[2]
    q_2_des = q_des[3]

    fig, axes = plt.subplots(4, 1, num="Errors of generalized coordinates", sharex=True) #figsize=(10, 5)

    axes[0].step(t, q_1 - q_1_des, color='b', linewidth=LINE_WIDTH, label=r"$\tilde{q}_1$")
    axes[1].step(t, q_2 - q_2_des, color='b', linewidth=LINE_WIDTH, label=r"$\tilde{q}_2$")
    axes[2].step(t, phi - phi_des, color='b', linewidth=LINE_WIDTH, label=r"$\tilde{\phi}$")
    axes[3].step(t, theta - theta_des, color='b', linewidth=LINE_WIDTH, label=r"$\tilde{\theta}$")

    axes[-1].set_xlabel("t [s]")
    axes[0].set_ylabel(r"$\tilde{q}_1 ~\mathrm{[rad]}$")
    axes[1].set_ylabel(r"$\tilde{q}_2 ~\mathrm{[rad]}$")
    axes[2].set_ylabel(r"$\tilde{\phi} ~\mathrm{[rad]}$")
    axes[3].set_ylabel(r"$\tilde{\theta} ~\mathrm{[rad]}$")

    for ax in axes:
        ax.set_xlim(t[0], t[-1])
        ax.grid()

    plt.tight_layout()
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif"
    # })

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')

    plt.show()


def plot_trajectories(
    t: NDArray,  # 1xN array with time instances
    q: NDArray,  # 4xN array with generalised coordinates at each time instance
    q_des: NDArray,  # 4x1 or 4xN array with desired "trajectories" (4x1: desired q, if 4xN: desired q trajectory)
    filepath: str = None,
) -> None:
    q_1 = q[0, :]
    phi = q[1, :]
    theta = q[2, :]
    q_2 = q[3, :]

    if q_des.ndim == 1:
        q_des = np.ones_like(q) * q_des[:, np.newaxis]

    q_1_des = q_des[0, :]
    phi_des = q_des[1, :]
    theta_des = q_des[2, :]
    q_2_des = q_des[3, :]

    fig, axes = plt.subplots(4, 1, num="Trajectories of generalized coordinates", sharex=True) #figsize=(10, 5)

    axes[0].step(t, q_1, color='b', linewidth=LINE_WIDTH, label=r"$q_1$")
    axes[0].step(t, q_1_des, color='b', linestyle='--', linewidth=LINE_WIDTH, label=r"$q_1^\mathrm{d}$")

    axes[1].step(t, q_2, color='b', linewidth=LINE_WIDTH, label=r"$q_2$")
    axes[1].step(t, q_2_des, color='b', linestyle='--', linewidth=LINE_WIDTH, label=r"$q_1^\mathrm{d}$")

    axes[2].step(t, phi, color='b', linewidth=LINE_WIDTH, label=r"$\phi$")
    axes[2].step(t, phi_des, color='b', linestyle='--', linewidth=LINE_WIDTH, label=r"$\phi^\mathrm{d}$")

    axes[3].step(t, theta, color='b', linewidth=LINE_WIDTH, label=r"\theta")
    axes[3].step(t, theta_des, color='b', linestyle='--', linewidth=LINE_WIDTH, label=r"\theta^\mathrm{d}$")

    axes[-1].set_xlabel("t [s]")
    axes[0].set_ylabel(r"$q_1 ~\mathrm{[rad]}$")
    axes[1].set_ylabel(r"$q_2 ~\mathrm{[rad]}$")
    axes[2].set_ylabel(r"$\phi ~\mathrm{[rad]}$")
    axes[3].set_ylabel(r"$\theta ~\mathrm{[rad]}$")

    for ax in axes:
        ax.set_xlim(t[0], t[-1])
        ax.grid()

    plt.tight_layout()
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "Times"
    # })

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')

    plt.show()
