from functools import partial
import numpy as np
from typing import Callable
from numpy.typing import NDArray

def rk4_step(ode_fun: Callable, x: NDArray, dt: NDArray) -> NDArray:
    k1 = ode_fun(x)
    k2 = ode_fun(x + dt * k1 / 2)
    k3 = ode_fun(x + dt * k2 / 2)
    k4 = ode_fun(x + dt * k3)

    x_next = x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * dt
    return x_next
