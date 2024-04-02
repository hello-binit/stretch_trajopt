import vtk
import time
import optas
import threading
import numpy as np
import casadi as cs
from pynput import keyboard
from stretch_traj_opt import Planner

############ SETUP ############

vis = optas.Visualizer(camera_position=[3, 3, 3])
planner = Planner(20, 2.0)

# the planned trajectory is initially just the robot's starting configuration
init_lift_height = 0.5
init_arm_extension = 0.25
init_wrist_yaw = np.deg2rad(0.0)
init_mobile_base_x = 0.0
init_mobile_base_y = 0.0
init_mobile_base_theta = 0.0
init_q = np.array([init_mobile_base_x, init_mobile_base_y, init_mobile_base_theta,
                   init_lift_height, init_arm_extension/4, init_arm_extension/4,
                   init_arm_extension/4, init_arm_extension/4, init_wrist_yaw])
robot_traj = init_q.reshape((9, 1))

############ TELEOP ############

print("Teleop Menu:")
print("  <up key>   : translate forwards along X axis")
print("  <down key> : translate backwards along X axis")
print("  <left key> : translate forwards along Y axis")
print("  <right key>: translate backwards along Y axis")
print("  w          : translate forwards along Z axis")
print("  s          : translate backwards along Z axis")
# print("  a          : rotate ccw around Z axis") # TODO: Add orientation
# print("  d          : rotate cw around Z axis")
print("  q          : Shutdown program")
print("Launching in 3 seconds...")
time.sleep(3)

keys = {'x+': False, 'x-': False, 'y+': False, 'y-': False,
        'z+': False, 'z-': False, 'yaw+': False, 'yaw-': False}

def on_press(key):
    if key == keyboard.Key.up:
        keys['x+'] = True
    elif key == keyboard.Key.down:
        keys['x-'] = True
    elif key == keyboard.Key.left:
        keys['y+'] = True
    elif key == keyboard.Key.right:
        keys['y-'] = True
    elif key == keyboard.KeyCode.from_char('w'):
        keys['z+'] = True
    elif key == keyboard.KeyCode.from_char('s'):
        keys['z-'] = True
    elif key == keyboard.KeyCode.from_char('a'):
        keys['yaw+'] = True
    elif key == keyboard.KeyCode.from_char('d'):
        keys['yaw-'] = True

def on_release(key):
    if key == keyboard.Key.up:
        keys['x+'] = False
    elif key == keyboard.Key.down:
        keys['x-'] = False
    elif key == keyboard.Key.left:
        keys['y+'] = False
    elif key == keyboard.Key.right:
        keys['y-'] = False
    elif key == keyboard.KeyCode.from_char('w'):
        keys['z+'] = False
    elif key == keyboard.KeyCode.from_char('s'):
        keys['z-'] = False
    elif key == keyboard.KeyCode.from_char('a'):
        keys['yaw+'] = False
    elif key == keyboard.KeyCode.from_char('d'):
        keys['yaw-'] = False

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release,
    suppress=False, # We want optas.Visualizer to catch 'q' to exit
)
listener.start()

############ MPC ############

def mpc_solve(lift_height, arm_extension, wrist_yaw, xyz_delta):
    # Initial configuration
    qc = np.array([lift_height, arm_extension/4, arm_extension/4,
                   arm_extension/4, arm_extension/4, wrist_yaw])
    qn = qc

    # Create desired path
    path = np.stack([
        np.linspace(0.0, xyz_delta[0], planner.T),
        np.linspace(0.0, xyz_delta[1], planner.T),
        np.linspace(0.0, xyz_delta[2], planner.T),
    ])

    # Transform path to end effector in global frame
    pn = cs.DM(planner.stretch.get_global_link_position("link_grasp_center", qc)).full()
    Rn = cs.DM(planner.stretch.get_global_link_rotation("link_grasp_center", qc)).full()
    for k in range(planner.T):
        path[:, k] = pn.flatten() + Rn @ path[:, k]

    # Plan!
    planner.reset(qc, qn, path)
    stretch_plan, mobile_base_plan = planner.plan()
    stretch_full_plan = cs.vertcat(mobile_base_plan, stretch_plan)

    return np.array(stretch_full_plan)

def handle_mpc(shutdown_flag):
    while not shutdown_flag.is_set():
        if any(keys.values()):
            # Pick where to plan from # TODO
            print('hey!1')
            print(keys)
            print(robot_traj.shape)

mpc_thread_shutdown_flag = threading.Event()
mpc_thread = threading.Thread(target=handle_mpc, args=(mpc_thread_shutdown_flag,))
mpc_thread.start()

############ VISUALIZE ############

# disable 'w'/'s' keys enabling/disabling wireframe vis - https://stackoverflow.com/a/69296623
class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
  def __init__(self, parent = None):
    self.AddObserver('CharEvent', self.OnChar)

  def OnChar(self, obj, event):
      if obj.GetInteractor().GetKeyCode() == "w":
          return
      super(MyInteractorStyle, obj).OnChar()
vis.iren.SetInteractorStyle(MyInteractorStyle())

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
        global robot_traj
        try:
            curr_q = robot_traj[:, 0]
            vis.actors.stop_adding_actors()
            self.curr_robot_vis = vis.robot(planner.stretch_full, curr_q)
            vis.actors.start_adding_actors()
            robot_traj = robot_traj[:, 1:]
        except:
            pass
        current = self.curr_robot_vis
        for actor in current:
            self.ren.AddActor(actor)

        # Render to window
        self.iren.GetRenderWindow().Render()

        # Reset
        self.prev = current

vis.animate_callbacks.append(CustomAnimationCallback(vis.iren, vis.ren))
vis.grid_floor()
vis.start()

############ SHUTDOWN ############

listener.stop()
mpc_thread_shutdown_flag.set()
mpc_thread.join()
