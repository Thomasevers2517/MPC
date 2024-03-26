import numpy as np
import control as ct

import physical_modelling.initialisation as init
import animation
from physical_modelling.forward_kinematics import calculate_link_endpoints


def set_initial_conditions():
    q_1_0 = 0
    phi_0 = np.deg2rad(2)
    theta_0 = np.deg2rad(2)
    q_2_0 = np.pi / 2

    q_1_dot_0 = 0
    phi_dot_0 = 0
    theta_dot_0 = 0
    q_2_dot_0 = 0

    q_0_ = init.initialize_q(q_1_0, phi_0, theta_0, q_2_0)
    q_dot_0_ = init.initialize_q_dot(q_1_dot_0, phi_dot_0, theta_dot_0, q_2_dot_0)

    return q_0_, q_dot_0_


if __name__ == "__main__":
    q_0, q_dot_0 = set_initial_conditions()
    physical_parameters = init.initialize_physical_parameters()

    link_ends = calculate_link_endpoints(physical_parameters, q_0, q_dot_0)
    # Dynamics.plot_mechanism_3d(link_ends)

    # sys = ct.NonlinearIOSystem(state_update, output, inputs=4, params=physical_parameters)
    # t = np.linspace(0, 5, 100, endpoint=False)
    # tau = np.zeros((len(t), 4))
    # x0 = np.concatenate([q_0, q_dot_0])
    # results = ct.input_output_response(sys, t, tau.T, x0)

    num_frames = 200
    q_1_test = np.linspace(0, np.pi / 2, num_frames)
    q_2_test = np.linspace(np.pi / 2, np.pi, num_frames)
    phi_test = np.linspace(0, np.pi / 1, num_frames)
    theta_test = np.linspace(0, 0, num_frames)
    q_test = np.array([q_1_test, phi_test, theta_test, q_2_test]).T
    q_dot_test = np.pi/10 * np.ones((num_frames, 4))
    link_ends_list = np.array([np.zeros_like(calculate_link_endpoints(physical_parameters, q_0))] * num_frames)
    for i in range(num_frames):
        link_ends_list[i] = calculate_link_endpoints(physical_parameters, q_test[i])

    animation.animate_mechanism_3d(link_ends_list)

    pass


