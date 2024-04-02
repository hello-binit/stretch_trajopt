import pickle
import argparse
from stretch_traj_opt import Planner
from optas.visualize import Visualizer


parser = argparse.ArgumentParser(description='mpc')
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
vis = Visualizer()
vis.robot_traj(planner.stretch_full, interpolated_solution, animate=True)
robot_traj = vis.animate_callbacks[0].traj
vis.animate_callbacks = []


class CustomAnimationCallback:
    def __init__(self, iren, ren):
        # Setup class attributes
        self.iren = iren
        self.prev = None
        self.ren = ren
        self.dt_ms = 100

    def start(self):
        self.iren.AddObserver("TimerEvent", self.callback)
        self.timer_id = self.iren.CreateRepeatingTimer(self.dt_ms)

    def callback(self, *args):
        # Remove previous actors
        if self.prev is not None:
            for actor in self.prev:
                self.ren.RemoveActor(actor)

        # Create new actors
        try:
            current = robot_traj.pop(0)
        except:
            current = None
        if current is not None:
            for actor in current:
                self.ren.AddActor(actor)

        # Render to window
        self.iren.GetRenderWindow().Render()

        # Reset
        self.prev = current

vis.animate_callbacks.append(CustomAnimationCallback(vis.iren, vis.ren))
vis.grid_floor()
vis.start()
