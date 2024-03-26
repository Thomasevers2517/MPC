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
    )

    # inertias = np.array([I_1, I_2, I_3, I_4, I_px, I_pz])
    # masses = np.array([m_1, m_2, m_3, m_4, m_p])
    # lengths = np.array([l_c1, l_c2, l_c3, l_c4, l_cp, L])
    #
    # physical_parameters = {'inertias': inertias, 'masses': masses, 'lengths': lengths}

    return physical_parameters


def initialize_q(q_1_0, phi_0, theta_0, q_2_0) -> NDArray:
    q_0 = np.array([q_1_0, phi_0, theta_0, q_2_0])
    return q_0


def initialize_q_dot(q_1_dot_0, phi_dot_0, theta_dot_0, q_2_dot_0) -> NDArray:
    q_dot_0 = np.array([q_1_dot_0, phi_dot_0, theta_dot_0, q_2_dot_0])
    return q_dot_0
