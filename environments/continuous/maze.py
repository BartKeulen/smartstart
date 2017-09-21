import math

import numpy as np
from Box2D import *


class Maze(object):
    EASY = 0
    MEDIUM = 1
    HARD = 2
    EXTREME = 3
    IMPOSSIBRUUHHH = 4

    def __init__(self, name, layout, wall_reset=False, visualizer=None, scale=5):
        self.name = name
        layout = np.asarray(layout)
        self.num_actions = 2

        grid_world = np.kron(layout, np.ones((scale, scale), dtype=layout.dtype))
        self.start_state = b2Vec2(tuple(np.where(grid_world == 2))[:, math.floor(scale ** 2 / 2)])
        self.goal_state = b2Vec2(tuple(np.where(grid_world == 3))[:, math.floor(scale ** 2 / 2)])
        grid_world[grid_world == 2] = 0
        grid_world[grid_world == 3] = 0
        # grid_world[tuple(start_state)] = 2
        # grid_world[tuple(goal_state)] = 3

        w, h = grid_world.shape

        wall_pos = []
        for i in range(len(grid_world)):
            for j in range(len(grid_world[0])):
                pos = (j * scale + scale/2 + 2, h - i * scale - scale/2 - 2)
                if layout[i][j] == 1:
                    wall_pos.append(pos)

        self.world = b2World(gravity=(0, 0))
        self.walls = []


        for pos in wall_pos:
            wall = self.world.CreateStaticBody(
                position=pos,
                shapes=b2PolygonShape(box=(scale/2, scale/2))
            )
            self.walls.append(wall)

        self.body = self.world.CreateDynamicBody(position=self.start_state,
                                                 linearDamping=0.5,
                                                 angularDamping=0.)
        box = self.body.CreateFixture(shape=b2CircleShape(pos=(0, 0), radius=scale/4),
                                      density=0.,
                                      friction=0.,
                                      restitution=0.)
