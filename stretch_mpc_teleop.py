import time
from stretch_traj_opt import Planner
from optas.visualize import Visualizer

from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersSources import vtkCylinderSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
)

start = time.time()
colors = vtkNamedColors()
cylinder = vtkCylinderSource()
cylinder.SetResolution(8)

cylinderMapper = vtkPolyDataMapper()
cylinderMapper.SetInputConnection(cylinder.GetOutputPort())


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
        current = []
        cylinderActor = vtkActor()
        cylinderActor.SetMapper(cylinderMapper)
        cylinderActor.GetProperty().SetColor(colors.GetColor3d('Tomato'))
        cylinderActor.RotateX(6 * (time.time() - start) % 30.0)
        cylinderActor.RotateY(-45.0)
        current.append(cylinderActor)
        for actor in current:
            self.ren.AddActor(actor)

        # Render to window
        self.iren.GetRenderWindow().Render()

        # Reset
        self.prev = current

vis = Visualizer()
# vis.actors.append(cylinderActor)
vis.animate_callbacks.append(CustomAnimationCallback(vis.iren, vis.ren))
vis.grid_floor()
vis.start()
