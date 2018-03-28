"""GridWorld Visualizer module

"""
import math
import pdb
from collections import deque

import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt

from smartstart.environments.environment import Visualizer

pygame.font.init()


class GridWorldVisualizer(Visualizer):
    """GridWorldVisualizer

    Render the gridworld an optional extra visualizations. Available
    visualizers:
        0.  console
        1.  live agent
        2.  value function
        3.  density

    Parameters
    ----------
    env : :obj:`~smartstart.environments.gridworld.GridWorld`
        GridWorld environment
    name : :obj:`str`
        visualizer name
    size : :obj:`tuple`
        size of each window in the visualizer (Default = (450, 450))
    fps : :obj:`int`
        frames per second for pygame

    Attributes
    ----------
    env : :obj:`~smartstart.environments.gridworld.GridWorld`
        GridWorld environment
    name : :obj:`str`
        visualizer name
    size : :obj:`tuple`
        size of each window in the visualizer (Default = (450, 450))
    fps : :obj:`int`
        frames per second for pygame
    screen : :obj:`pygame.Surface`
        pygame surface for rendering
    clock : :obj:`pygame.Clock`
        pygame clock
    grid : :obj:`np.ndarray`
        GridWorld grid
    colors : :obj:`dict`
        colors for rendering the GridWorld
    messages : :obj:`collections.deque`
        deque with messages to show in the console window (maxlen=29)
    active_visualizers : :obj:`set`
        contain the visualizers to be used
    """

    def __init__(self, env, name="GridWorld", size=None, fps=60):
        env.visualizer = self
        self.env = env
        self.name = name
        if size is None:
            size = (450, 450)
        self.spacing = 10
        self.size = size
        self.fps = fps

        self.screen = None
        self.clock = None
        self.grid = None

        self.colors = {
            'background': (0, 0, 0, 0),
            'start': (255, 255, 255, 255),
            'wall': (51, 107, 135, 255),
            'goal': (0, 255, 0, 255),
            'subgoal': (255, 255, 0, 255),
            'agent': (144, 175, 197, 255),
            'path': (255, 255, 255, 255)
        }

        self.messages = deque(maxlen=29)
        self.active_visualizers = set()

    def add_visualizer(self, *args):
        """Add visualizer

        Parameters
        ----------
        *args :
            visualizers to add, see class attributes for available visualizers
        """
        for arg in args:
            if arg == self.ALL:
                self.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                                    GridWorldVisualizer.VALUE_FUNCTION,
                                    GridWorldVisualizer.DENSITY,
                                    GridWorldVisualizer.CONSOLE)
            else:
                self.active_visualizers.add(arg)

    def render(self, value_map=None, density_map=None, message=None, close=False, fp=None):
        """Render the current state of the GridWorld

        Parameters
        ----------
        value_map : :obj:`np.ndarray`
             (Default value = None)
        density_map : :obj:`np.ndarray`
             (Default value = None)
        message : :obj:`str`
             (Default value = None)
        close : :obj:`bool`
             (Default value = False)

        Returns
        -------
        :obj:`bool`
            True if the agent has to stop rendering
        """
        # Create screen on first call
        if self.screen is None:
            w, h = 1, 1
            if len(self.active_visualizers) > 1:
                w = 2
                if len(self.active_visualizers) > 2:
                    h = 2
            # if w == 1 and h == 1:
            #     size = (self.size[0] * w, self.size[1] * h)
            # else:
            #
            size = (self.size[0] * w, self.size[1] * h)
            self.screen = pygame.display.set_mode(size, 0, 32)
            pygame.display.set_caption(self.name)
            self.clock = pygame.time.Clock()

        # Check for events
        if self.screen is not None:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    close = True

        # Close the screen
        if close:
            if self.screen is not None:
                pygame.display.quit()
                self.screen = None
                return False

        # Fill background with black
        self.screen.fill(self.colors['background'])

        # Render agent, value map, density map and console
        self.grid = self.env.get_grid()
        positions = [(1, 1), (0, 1), (1, 0), (0, 0)]
        if self.LIVE_AGENT in self.active_visualizers:
            pos = positions.pop()
            self._render_grid(pos=pos)
            self._render_borders(pos=pos)
            self._render_walls(pos=pos)
            self._render_elements("goal", "subgoal", "start", "agent", pos=pos)
            # self._render_map(grid, pos=positions.pop())
        if self.VALUE_FUNCTION in self.active_visualizers and value_map is not None:
            pos = positions.pop()
            self._render_grid(pos=pos)
            self._render_borders(pos=pos)
            self._render_walls(pos=pos)
            self._render_value_map(value_map, pos=pos)
            self._render_elements("goal", "subgoal", "start", pos=pos)
        if self.DENSITY in self.active_visualizers and density_map is not None:
            pos = positions.pop()
            self._render_grid(pos=pos)
            self._render_borders(pos=pos)
            self._render_walls(pos=pos)
            self._render_value_map(density_map, pos=pos)
            self._render_elements("goal", "subgoal", "start", pos=pos)
        if self.CONSOLE in self.active_visualizers:
            pos = positions.pop()
            self._render_borders(pos=pos)
            self._render_console(pos=pos, message=message)

        pygame.display.flip()
        self.clock.tick(self.fps)

        if fp is not None:
            pygame.image.save(self.screen, fp)

        return True

    def _render_grid(self, pos=(0, 0)):
        grid_h, grid_w = self.grid.shape
        w, h = self.size

        offset_left, offset_top = self._get_offset(pos)
        scale_w, scale_h, overshoot_w, overshoot_h = self._get_scale_overshoot()
        border_left, border_top, border_right, border_bottom = self._get_borders(overshoot_w, overshoot_h)

        color = (211, 211, 211, 255)
        for y in range(grid_h):
            start = np.array([border_left + offset_left, y * scale_h + border_top + offset_top])
            end = np.array([w - border_left + offset_left, y * scale_h + border_top + offset_top])
            pygame.draw.line(self.screen, color, start, end, 1)

        for x in range(grid_w):
            start = np.array([x * scale_w + offset_left + border_left, border_top + offset_top])
            end = np.array([x * scale_w + offset_left + border_left, h - border_top + offset_top])
            pygame.draw.line(self.screen, color, start, end, 1)

    def _render_borders(self, pos=(0, 0)):
        w, h = self.size
        offset_left, offset_top = self._get_offset(pos)
        scale_w, scale_h, overshoot_w, overshoot_h = self._get_scale_overshoot()
        border_left, border_top, border_right, border_bottom = self._get_borders(overshoot_w, overshoot_h)

        color = self.colors["wall"]
        if border_left > 0:
            vertices = np.array([[0 + offset_left, 0 + offset_top], [border_left + offset_left, 0 + offset_top],
                                 [border_left + offset_left, h + offset_top], [0 + offset_left, h + offset_top]])
            pygame.draw.polygon(self.screen, color, vertices)

        if border_top > 0:
            vertices = np.array([[0 + offset_left, 0 + offset_top], [w + offset_left, 0 + offset_top],
                                 [w + offset_left, border_top + offset_top],
                                 [0 + offset_left, border_top + offset_top]])
            pygame.draw.polygon(self.screen, color, vertices)

        if border_right > 0:
            vertices = np.array([[w - border_right + offset_left, 0 + offset_top], [w + offset_left, 0 + offset_top],
                                 [w + offset_left, h + offset_top], [w - border_right + offset_left, h + offset_top]])
            pygame.draw.polygon(self.screen, color, vertices)

        if border_bottom > 0:
            vertices = np.array([[0 + offset_left, h - border_bottom + offset_top],
                                 [w + offset_left, h - border_bottom + offset_top], [w + offset_left, h + offset_top],
                                 [0 + offset_left, h + offset_top]])
            pygame.draw.polygon(self.screen, color, vertices)

    def _render_elements(self, *args, pos=(0, 0)):
        """Render a single element

        available elements are:
            * 'goal'
            * 'start'
            * 'agent'

        Parameters
        ----------
        *args :
            elements to render
        pos : :obj:`tuple`
            position of the window in the visualizer (Default value = (0, 0))
        """
        offset_left, offset_top = self._get_offset(pos)
        scale_w, scale_h, overshoot_w, overshoot_h = self._get_scale_overshoot()
        border_left, border_top, border_right, border_bottom = self._get_borders(overshoot_w, overshoot_h)

        for element in args:
            res = self._get_element(element)
            if isinstance(res, list):
                for x, y in res:
                    vertices = np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]])
                    vertices[:, 0] = vertices[:, 0] * scale_w + border_left + offset_left
                    vertices[:, 1] = vertices[:, 1] * scale_h + border_top + offset_top

                    color = self.colors[element]
                    pygame.draw.polygon(self.screen, color, vertices)
            else:
                x, y = res
                pos = np.array([x, y]) + 1/2
                pos *= np.array([scale_w, scale_h])
                pos += np.array([border_left + offset_left, border_top + offset_top])
                pos = np.asarray(pos, dtype=np.int)

                color = self.colors[element]
                radius = int(min(scale_w, scale_h) / 2)
                pygame.draw.circle(self.screen, color, tuple(pos), radius)

    def _get_element(self, element):
        """Returns the position of the element

        available elements are:
            * 'goal'
            * 'start'
            * 'agent'

        Parameters
        ----------
        element : :obj:`str`
            element to return the position from

        Returns
        -------
        :obj:`int`
            x coordinate
        :obj:`int`
            y coordinate
        """
        if element == "start":
            y, x = self.env.start_state
        elif element == "goal":
            y, x = self.env.goal_state
        elif element == "agent":
            y, x = self.env.state
        elif element == "subgoal":
            if self.env.subgoal_state is not None:
                if isinstance(self.env.subgoal_state, list):
                    return self.env.subgoal_state
                y, x = self.env.subgoal_state
            else:
                y, x = -10, -10
        else:
            raise NotImplementedError

        return x, y

    def _render_walls(self, pos=(0, 0)):
        """Render the walls

        renders all the walls as defined in the GridWorld. Adds walls to
        remaining spaces on the top, right, bottom and left.

        Parameters
        ----------
        pos : :obj:`tuple`
             position of the window in the visualizer (Default value = (0, 0))
        """
        grid_h, grid_w = self.grid.shape
        w, h = self.size

        offset_left, offset_top = self._get_offset(pos)
        scale_w, scale_h, overshoot_w, overshoot_h = self._get_scale_overshoot()
        border_left, border_top, border_right, border_bottom = self._get_borders(overshoot_w, overshoot_h)

        color = self.colors["wall"]
        for y in range(grid_h):
            for x in range(grid_w):
                cell_type = self.grid[y, x]
                if cell_type == 1:
                    vertices = np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]])
                    vertices[:, 0] = vertices[:, 0] * scale_w + border_left + offset_left
                    vertices[:, 1] = vertices[:, 1] * scale_h + border_top + offset_top

                    pygame.draw.polygon(self.screen, color, vertices)

    def _render_value_map(self, value_map, pos=(0, 0)):
        """Render value map or density map

        Normalizes the values in the value_map and turns them into a
        :class:`matplotlib.colors.Colormap`. The colors are then scaled to
        values between 0 and 255 and used for rendering.

        Walls will be rendered in the wall color.

        Parameters
        ----------
        value_map : :obj:`np.ndarray`
            value function map or density map
        pos : :obj:`tuple`
             position of the window in the visualizer (Default value = (0, 0))
        """
        grid_h, grid_w = self.grid.shape

        offset_left, offset_top = self._get_offset(pos)
        scale_w, scale_h, overshoot_w, overshoot_h = self._get_scale_overshoot()
        border_left, border_top, border_right, border_bottom = self._get_borders(overshoot_w, overshoot_h)

        # Normalize map
        # if np.sum(value_map) != 0.:
        #     value_map /= np.max(value_map)
        scale = 40
        value_map = np.clip(value_map, 0, scale) / scale
        cmap = plt.get_cmap('hot')
        rgba_img = cmap(value_map) * 255

        for y in range(grid_h):
            for x in range(grid_w):
                cell_type = self.grid[y, x]
                vertices = np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]])
                vertices[:, 0] = vertices[:, 0] * scale_w + border_left + offset_left
                vertices[:, 1] = vertices[:, 1] * scale_h + border_top + offset_top

                color = tuple(rgba_img[y, x])

                if cell_type == 1:
                    color = self.colors["wall"]

                pygame.draw.polygon(self.screen, color, vertices)

    def _render_console(self, pos, message=None):
        """Render console window

        Parameters
        ----------
        pos : :obj:`tuple`
            position of the window in the visualizer (Default value = (0, 0))
        message : :obj:`str`
             message to add to the console (Default value = None)
        """
        w, h = self.size
        offset_left = pos[0] * (w + self.spacing) + 5
        offset_top = pos[1] * (h + self.spacing) + 5

        if message is not None:
            self.messages.append(message)

        if not self.messages:
            return

        basic_font = pygame.font.SysFont('Sans', 15)
        for i, message in enumerate(self.messages):
            text = basic_font.render(message, True, (255, 255, 255, 255))
            self.screen.blit(text, (offset_left, offset_top + i * 15))

    def _get_offset(self, pos):
        """Return the left and top offset of a window

        To create spacing between the windows the right windows need an
        offset on the left and the bottom windows on the top.

        Parameters
        ----------
        pos : :obj:`tuple`
            position of the window in the visualizer (Default value = (0, 0))

        Returns
        -------
        :obj:`int`
            offset left
        :obj:`int`
            offset top
        """
        w, h = self.size
        offset_left = pos[0] * w
        offset_top = pos[1] * h
        return offset_left, offset_top

    def _get_scale_overshoot(self):
        """Calculate the scale and overshoot of GridWorld

        The GridWorld often does not scale correctly with the available window
        size. The overshoot is the remaining portion when the GridWorld is
        scaled to fit the biggest portion of the window.

        Returns
        -------
        :obj:`int`
            scale width
        :obj:`int`
            scale height
        :obj:`int`
            overshoot width
        :obj:`int`
            overshoot height
        """
        grid_h, grid_w = self.grid.shape
        w, h = self.size
        overshoot_w = (w - self.spacing) % grid_w
        overshoot_h = (h - self.spacing) % grid_h
        scale_w = int((w - overshoot_w) / grid_w)
        scale_h = int((h - overshoot_h) / grid_h)
        return scale_w, scale_h, overshoot_w, overshoot_h

    def _get_borders(self, overshoot_w, overshoot_h):
        """Returns size of the borders

        The border divide the overshoot equally over each side.

        Parameters
        ----------
        overshoot_w : :obj:`int`
            overshoot width
        overshoot_h : :obj:`int`
            overshoot height

        Returns
        -------
        :obj:`float`
            border left
        :obj:`float`
            border top
        :obj:`float`
            border right
        :obj:`float`
            border bottom
        """
        border_left = math.floor(overshoot_w / 2)
        border_right = overshoot_w - border_left

        border_top = math.floor(overshoot_h / 2)
        border_bottom = overshoot_h - border_top

        border_left += self.spacing/2
        border_right += self.spacing/2
        border_top += self.spacing/2
        border_bottom += self.spacing/2

        return border_left, border_top, border_right, border_bottom
