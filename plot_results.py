import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray

LINE_WIDTH = 1


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

    axes[0].plot(t, q_1 - q_1_des, color='b', linewidth=LINE_WIDTH, label=r"$\tilde{q}_1$")
    axes[1].plot(t, q_2 - q_2_des, color='b', linewidth=LINE_WIDTH, label=r"$\tilde{q}_2$")
    axes[2].plot(t, phi - phi_des, color='b', linewidth=LINE_WIDTH, label=r"$\tilde{\phi}$")
    axes[3].plot(t, theta - theta_des, color='b', linewidth=LINE_WIDTH, label=r"$\tilde{\theta}$")

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
        plt.savefig(filepath)

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

    axes[0].plot(t, q_1, color='b', linewidth=LINE_WIDTH, label=r"$q_1$")
    axes[0].plot(t, q_1_des, color='b', linestyle='--', linewidth=LINE_WIDTH, label=r"$q_1^\mathrm{d}$")

    axes[1].plot(t, q_2, color='b', linewidth=LINE_WIDTH, label=r"$q_2$")
    axes[1].plot(t, q_2_des, color='b', linestyle='--', linewidth=LINE_WIDTH, label=r"$q_1^\mathrm{d}$")

    axes[2].plot(t, phi, color='b', linewidth=LINE_WIDTH, label=r"$\phi$")
    axes[2].plot(t, phi_des, color='b', linestyle='--', linewidth=LINE_WIDTH, label=r"$\phi^\mathrm{d}$")

    axes[3].plot(t, theta, color='b', linewidth=LINE_WIDTH, label=r"\theta")
    axes[3].plot(t, theta_des, color='b', linestyle='--', linewidth=LINE_WIDTH, label=r"\theta^\mathrm{d}$")

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
        plt.savefig(filepath)

    plt.show()
