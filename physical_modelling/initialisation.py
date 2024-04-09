import numpy as np

from typing import Dict, Tuple
from numpy.typing import NDArray


def initialize_physical_parameters() -> Dict[str, float]:
    physical_parameters = dict(
        I_1=0.0017,     # [kg m^2]
        I_2=0.00001,    # [kg m^2]
        I_3=0.00001,    # [kg m^2]
        I_4=0.0014,     # [kg m^2]
        I_px=0.0017,    # [kg m^2]
        I_pz=0.000003,  # [kg m^2]
        m_1=0.126,      # [kg]
        m_2=0.08535,    # [kg]
        m_3=0.063,      # [kg]
        m_4=0.121,      # [kg]
        m_p=0.125,      # [kg]
        l_c1=0.047,     # [m]
        l_c2=0.069,     # [m]
        l_c3=0.062,     # [m]
        l_c4=0.045,     # [m]
        l_cp=0.155,     # [m]
        L=0.127,        # [m]
        g_0=9.81        # [m/s^2]
    )

    return physical_parameters


# initialise simulation time instances
def generate_t_ts(sim_dt, sim_duration):
    sim_length = int(sim_duration / sim_dt)
    t_ts = sim_dt * np.arange(sim_length)

    return t_ts, sim_length


def initialize_q(q_1_0, phi_0, theta_0, q_2_0) -> NDArray:
    q_0 = np.array([q_1_0, phi_0, theta_0, q_2_0])
    return q_0


def initialize_q_dot(q_1_dot_0, phi_dot_0, theta_dot_0, q_2_dot_0) -> NDArray:
    q_dot_0 = np.array([q_1_dot_0, phi_dot_0, theta_dot_0, q_2_dot_0])
    return q_dot_0


def set_initial_conditions(q_1_0, phi_0, theta_0, q_2_0, q_1_dot_0, phi_dot_0, theta_dot_0, q_2_dot_0):
    q_0_ = initialize_q(q_1_0, phi_0, theta_0, q_2_0)
    q_dot_0_ = initialize_q_dot(q_1_dot_0, phi_dot_0, theta_dot_0, q_2_dot_0)

    return q_0_, q_dot_0_

