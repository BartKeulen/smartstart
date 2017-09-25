import numpy as np

from smartexploration.environments.gridworldvisualizer import GridWorldVisualizer


class MazeVisualizer(GridWorldVisualizer):

    def __init__(self, env, name="Maze", size=None, fps=60):
        super(MazeVisualizer, self).__init__(env, name, size, fps)

    def _get_element(self, element):
        if element == "start":
            y, x = self.env.start_state
        elif element == "goal":
            y, x = self.env.goal_state
        elif element == "agent":
            x, y = np.asarray(self.env.body.position) / self.env.scale - np.array([1/2, -1/2])
            y = self.env.h - y
        else:
            raise NotImplementedError

        return x, y

