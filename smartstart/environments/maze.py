"""Maze module

2D point-mass implementation of GridWorld. The environment has continuous
states and discrete actions.
"""
import numpy as np
from Box2D import *

from smartstart.environments.gridworld import GridWorld


class Maze(GridWorld):
    """Maze

    Parameters
    ----------
    name : :obj:`str`
        name is being prefixed by class name
    layout : double :obj:`list` of :obj:`int` or :obj:`np.ndarray`
        layout of the Maze


    """

    def __init__(self, name, layout, wall_reset=False, scale=5, dt=0.05):
        super(Maze, self).__init__(name, layout, wall_reset=wall_reset,
                                   scale=scale)
        self.name = self.__class__.__name__ + name
        self.num_actions = 2

        # Create world
        self.scale = scale
        self.world = b2World(gravity=(0, 0))

        # Create walls
        for i in range(self.w):
            for j in range(self.h):
                pos = ((i + 1/2) * scale, (self.h - j - 1/2) * scale)
                if self.grid_world[j][i] == 1:
                    self.world.CreateStaticBody(
                        position=pos,
                        shapes=b2PolygonShape(box=(scale / 2, scale / 2))
                    )

                elif self.grid_world[j][i] == 2:
                    self.box2d_start = pos

                elif self.grid_world[j][i] == 3:
                    self.box2d_goal = pos

        self.world.CreateStaticBody(
            position=(-scale/2, self.h*scale/2),
            shapes=b2PolygonShape(box=(scale/2, self.h*scale/2))
        )

        self.world.CreateStaticBody(
            position=(self.w*scale/2, (self.h + 1/2)*scale),
            shapes=b2PolygonShape(box=(self.w*scale/2, scale/2))
        )

        self.world.CreateStaticBody(
            position=((self.w + 1/2)*scale, self.h*scale/2),
            shapes=b2PolygonShape(box=(scale/2, self.h*scale/2))
        )

        self.world.CreateStaticBody(
            position=(self.w*scale/2, -scale/2),
            shapes=b2PolygonShape(box=(self.w*scale/2, scale/2))
        )

        # Create agent
        self.body = self.world.CreateDynamicBody(position=tuple(self.box2d_start),
                                                 linearDamping=0.5,
                                                 angularDamping=0.)
        self.body.CreateFixture(shape=b2CircleShape(pos=(0, 0), radius=scale/2),
                                density=0.,
                                friction=0.,
                                restitution=0.)

        # Set agent dynamics
        c = self.body.linearDamping
        m = self.body.mass

        A = np.eye(4)
        A[0, 2] = dt
        A[1, 3] = dt
        A[2, 2] -= c / m * dt
        A[3, 3] -= c / m * dt
        B = np.zeros((4, 2))
        B[2, 0] = force_scale * dt / m
        B[3, 1] = force_scale * dt / m
        self.A, self.B = A, B
        self.dt = dt

        self._u_high = np.ones(2)
        self._max_speed = np.sqrt(50)

    def reset(self, start_state=None):
        """

        Parameters
        ----------
        start_state :
             (Default value = None)

        Returns
        -------

        """
        if start_state is None:
            self.body.position = b2Vec2(self.box2d_start)
        else:
            self.body.position = b2Vec2(start_state)
        self.body.linearVelocity = b2Vec2(0, 0)
        self.body.angularVelocity = 0.
        self.body.angle = 0.
        return self._get_obs()

    def step(self, action):
        """

        Parameters
        ----------
        action :
            

        Returns
        -------

        """
        u = np.clip(action, -self._u_high, self._u_high)

        x = np.concatenate((self.body.position, self.body.linearVelocity))
        x_new = self.dynamics(x, u)
        self.body.position = x_new[:2]
        self.body.linearVelocity = x_new[2:]

        if self.goal_state is not None:
            done = (np.linalg.norm(self.body.position - self.box2d_goal) < self.scale/2)
        else:
            done = False
        reward = 1 if done else 0

        self.world.Step(self.dt, 10, 10)

        return self._get_obs(), reward, done, {}

    def _get_obs(self, x=None):
        """

        Parameters
        ----------
        x :
             (Default value = None)

        Returns
        -------

        """
        if x is None:
            x = np.concatenate((self.body.position, self.body.linearVelocity))
        return x

    def dynamics(self, x, u):
        """

        Parameters
        ----------
        x :
            
        u :
            

        Returns
        -------

        """
        u = np.clip(u, -self._u_high, self._u_high)

        x_new = self.A.dot(x) + self.B.dot(u)

        return x_new.copy()

if __name__ == "__main__":
    from smartstart.environments import MazeVisualizer
    import pygame
    from pygame.locals import *

    env = Maze.generate(Maze.IMPOSSIBRUUHHH)
    vis = MazeVisualizer(env)
    vis.add_visualizer(MazeVisualizer.LIVE_AGENT)
    env.visualizer = vis
    obs = env.reset()

    render = True
    done = False
    while not done:
        if render:
            render = env.render()

        action = np.zeros(2)
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    action = np.array([0, 1])
                elif event.key == K_RIGHT:
                    action = np.array([1, 0])
                elif event.key == K_DOWN:
                    action = np.array([0, -1])
                elif event.key == K_LEFT:
                    action = np.array([-1, 0])

        obs, reward, done, _ = env.step(action)

    print("FINISHED!!")