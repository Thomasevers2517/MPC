import numpy as np

from functools import partial
from typing import Callable, Dict, Optional, Tuple
from numpy.typing import NDArray

from dynamics import discrete_forward_dynamics
from forward_kinematics import extended_forward_kinematics, calculate_link_endpoints


def simulate_robot(
    physical_parameters: dict,
    t_ts: NDArray,
    discrete_forward_dynamics_fn: Optional[Callable] = None,
    q_0: NDArray = np.array([0.0, 0.0]),
    q_dot_0: NDArray = np.array([0.0, 0.0]),
    tau_ext_ts: Optional[NDArray] = None,
    q_des_ts: Optional[NDArray] = None,
    q_dot_des_ts: Optional[NDArray] = None,
    q_ddot_des_ts: Optional[NDArray] = None,
    ctrl_ff: Callable = lambda q, q_dot, q_des, q_dot_des, q_ddot_des: np.zeros((4,)),
    ctrl_fb: Callable = lambda q, q_dot, q_des, q_dot_des: np.zeros((4,)),
) -> Dict[str, NDArray]:
    if discrete_forward_dynamics_fn is None:
        discrete_forward_dynamics_fn = partial(discrete_forward_dynamics, physical_parameters)

    if tau_ext_ts is None:
        tau_ext_ts = np.zeros((t_ts.shape[0], 2))

    if q_des_ts is None:
        q_des_ts = np.zeros((t_ts.shape[0], 4))

    if q_dot_des_ts is None:
        q_dot_des_ts = np.zeros((t_ts.shape[0], 4))

    if q_ddot_des_ts is None:
        q_ddot_des_ts = np.zeros((t_ts.shape[0], 4))

    num_time_steps = t_ts.shape[0]
    dt = t_ts[1] - t_ts[0]

    step_simulator_fn = partial(step_simulator, physical_parameters, discrete_forward_dynamics_fn, ctrl_ff, ctrl_fb)

    input_ts = dict(t_ts=t_ts, tau_ext_ts=tau_ext_ts, q_des_ts=q_des_ts,
                    q_dpt_des_ts=q_dot_des_ts, q_ddot_des_ts=q_ddot_des_ts)

    carry = dict(t=t_ts[0] - dt, q=q_0, q_dot=q_dot_0)

    _sim_ts = []
    for time_idx in range(num_time_steps):
        input = {k: v[time_idx] for k, v in input_ts.items()}

        carry, step_data = step_simulator_fn(carry, input)
        _sim_ts.append(step_data)

    sim_ts = {k: np.stack([step_data[k] for step_data in _sim_ts]) for k in _sim_ts[0]}

    return sim_ts


def step_simulator(
    physical_parameters: dict,
    discrete_forward_dynamics_fn: Callable,
    ctrl_ff: Callable,
    ctrl_fb: Callable,
    carry: Dict[str, NDArray],
    input: Dict[str, NDArray],
) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
    # extract the current state from the carry dictionary
    t_curr = carry["t"]
    q_curr = carry["q"]
    q_dot_curr = carry["q_d"]

    # compute time step
    dt = input["t_ts"] - t_curr

    # evaluate feedforward and feedback controllers
    q_des = input["q_des_ts"]
    q_dot_des = input["q_dot_des_ts"]
    q_ddot_des = input["q_ddot_des_ts"]
    tau_ff = ctrl_ff(q_curr, q_dot_curr, q_des, q_dot_des,  q_ddot_des)
    tau_fb = ctrl_fb(q_curr, q_dot_curr, q_des, q_dot_des)

    # compute total torque
    tau = input["tau_ext_ts"] + tau_ff + tau_fb

    # evaluate the dynamics
    q_next, q_dot_next, q_ddot = discrete_forward_dynamics_fn(dt, q_curr, q_dot_curr, tau)

    # evaluate forward kinematics at the current time step
    link_endpoints_current = calculate_link_endpoints(physical_parameters, q_curr)
    kinematics_dict = extended_forward_kinematics(physical_parameters, q_curr, q_dot_curr, q_ddot)
    x_curr = kinematics_dict['x']
    x_dot_curr = kinematics_dict['x_dot']
    x_ddot = kinematics_dict['x_ddot']

    # save the current state and the state transition data
    step_data = dict(
        t_ts=carry["t"],
        q_ts=q_curr,
        q_dot_ts=q_dot_curr,
        q_ddot_ts=q_ddot,
        x_ts=x_curr,
        x_d_ts=x_dot_curr,
        x_dd_ts=x_ddot,
        tau_ts=tau,
        tau_ff_ts=tau_ff,
        tau_fb_ts=tau_fb,
    )

    # update the carry array
    carry = dict(t=input["t_ts"], q=q_next, q_dot=q_dot_next)

    return carry, step_data