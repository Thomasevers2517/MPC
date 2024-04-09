Couldn't get the custom simulation and controls to work, so the following files and functions are **not** in use:
- `simulation.py`:
  - `simulate_robot`
  - `_step_simulator`
- `linearisation.py`: entire file
- `dynamics.py`:
  - `discrete_forward_dynamics`
  - `continuous_linear_state_space_representation`: not even implemented
  - `continuous_closed_loop_linear_state_space_representation`: not even implemented
- `controllers.py`:
  - `lqr`
- `symbolic.py`: entire file, but it was useful to verify the paper's calculations (beware: incredibly slow, caused by the substitution and simplification)


### TODO:
- add $\phi$ and $\theta$ as disturbances to the system
- add option for trajectory following (now controlled to set point)
  - difficulty: A, B, C, D matrices are given for one equilibrium point only in the paper, so we have to implement linearisation around any equlibrium (with the symbolic file or the control library's linearisation - see issue #1)
- implement MPC


### Issues:
1. linearizing the nonlinear system with the control library doesn't yield the same results as the paper (currently using the matrices in the paper)
2. animation is slow (update function can't update the graphics fast enough, I believe)
3. singularity in kinematics doesn't allow the mechanism to reach a state where the two middle bars form a straight line