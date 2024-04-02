import pickle
import argparse
import numpy as np
from optas.visualize import Visualizer
from stretch_traj_opt import Planner

parser = argparse.ArgumentParser(description='generate solution.')
parser.add_argument("--load-dir", nargs="?", type=str, const="",
                    help="load soln from given directory.")
parser.add_argument("--time-steps", type=int, default=200,
                    help="The number of minutes of data to save.")
parser.add_argument("--total-time", type=float, default=20.0,
                    help="The number of minutes of data to save.")
args, _ = parser.parse_known_args()

with open(f'{args.load_dir}/path.pkl', 'rb') as f:
    path = pickle.load(f)

with open(f'{args.load_dir}/path_actual.pkl', 'rb') as f:
    path_actual = pickle.load(f)

with open(f'{args.load_dir}/interpolated_solution.pkl', 'rb') as f:
    interpolated_solution = pickle.load(f)

planner = Planner(args.time_steps, args.total_time)
vis = Visualizer(camera_position=[3, 3, 3])

for i in range(len(path[0, :])):
    vis.sphere(position=path[:, i], radius=0.01, rgb=[1, 0, 0])
    vis.sphere(position=path_actual[:, i], radius=0.01, rgb=[0, 1, 0])

vis.grid_floor()
vis.robot_traj(planner.stretch_full, np.array(interpolated_solution),
                animate=True, duration=planner.Tmax)
vis.start()
