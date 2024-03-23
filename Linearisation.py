import numpy as np
def continuous_linear_dynamics(q, dq, tau, M = None, C = None, g=None, calc_row=None, calc_drow=None):
    row = calc_row(q)
    d_row = calc_drow(q, dq)
    
    dq = dq
    ddq = np.linalg.inv(M(row)) @ (tau - C(row, d_row) - g(row))
    
