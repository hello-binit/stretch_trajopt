import pickle
import numpy as np
import casadi as cs
from stretch_traj_opt import Planner, get_three_point_path

planner = Planner(200, 20.0)

# Initial Arm Configuration
lift_height = 0.5
arm_extention = 0.25
wrist_yaw = np.deg2rad(0.0)
qc = np.array([lift_height, arm_extention/4, arm_extention/4,
                        arm_extention/4, arm_extention/4, wrist_yaw])
qn = qc

pn = cs.DM(planner.stretch.get_global_link_position("link_grasp_center", qc)).full()
Rn = cs.DM(planner.stretch.get_global_link_rotation("link_grasp_center", qc)).full()
t  = cs.DM(planner.t_).full()

path = get_three_point_path(planner.T,
    -1.0, -1.0, -0.2,
     0.0, -1.0,  0.0,
     0.0,  0.0,  0.0,
)

# Transform path to end effector in global frame
for k in range(planner.T):
    path[:, k] = pn.flatten() + Rn @ path[:, k]

planner.reset(qc, qn, path)
stretch_plan, mobile_base_plan = planner.plan()
stretch_full_plan = cs.vertcat(mobile_base_plan, stretch_plan)

path_actual = np.zeros((3, planner.T))
for k in range(planner.T):
    q_sol = np.array(stretch_full_plan[:, k])
    pn = cs.DM(planner.stretch_full.get_global_link_position("link_grasp_center",
                                                                q_sol)).full()
    path_actual[:, k] = pn.flatten()

# Optionally: interpolate between timesteps for smoother animation
timestep_mult = 2 # 1 means no interpolation
original_timesteps = np.linspace(0, 1, stretch_full_plan.size2())
interpolated_timesteps = np.linspace(0, 1,
                                        timestep_mult * stretch_full_plan.size2())
interpolated_solution = cs.DM.zeros(stretch_full_plan.size1(),
                                    len(interpolated_timesteps))
for i in range(stretch_full_plan.size1()):
    interpolated_solution[i, :] = cs.interp1d(original_timesteps,
                                                stretch_full_plan[i, :].T,
                                                interpolated_timesteps)
interpolated_solution = np.array(interpolated_solution)

with open('path.pkl', 'wb') as f:
    pickle.dump(path, f)

with open('path_actual.pkl', 'wb') as f:
    pickle.dump(path_actual, f)

with open('interpolated_solution.pkl', 'wb') as f:
    pickle.dump(interpolated_solution, f)
