import math
from collections import deque

import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt

pygame.font.init()


class GridWorldVisualizer(object):
    CONSOLE = 0
    LIVE_AGENT = 1
    VALUE_FUNCTION = 2
    DENSITY = 3

    def __init__(self, name="GridWorld", size=None, fps=60):
        self.name = name
        if size is None:
            size = (450, 450)
        self.spacing = 10
        self.size = size
        self.fps = fps

        self.screen = None
        self.clock = None

        self.colors = {
            '0': (0, 0, 0, 0),
            '2': (255, 255, 255, 255),
            '1': (51, 107, 135, 255),
            '3': (0, 255, 0, 255),
            '4': (144, 175, 197, 255),
            'path': (255, 255, 255, 255)
        }

        self.messages = deque(maxlen=29)
        self.active_visualizers = set()

    def add_visualizer(self, *args):
        for arg in args:
            self.active_visualizers.add(arg)

    def render(self, grid, value_map=None, density_map=None, message=None, close=False):
        # Create screen on first call
        if self.screen is None:
            w, h = 1, 1
            if len(self.active_visualizers) > 1:
                w = 2
                if len(self.active_visualizers) > 2:
                    h = 2
            size = (self.size[0] * w + self.spacing, self.size[1] * h + self.spacing)
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
        self.screen.fill(self.colors['0'])

        # Render dividers
        w, h = self.size
        hor_divider = np.array([[0, h], [w*2 + self.spacing, h], [w*2 + self.spacing, h + self.spacing], [0, h + self.spacing]])
        pygame.draw.polygon(self.screen, self.colors['1'], hor_divider)
        ver_divider = np.array([[w, 0], [w + self.spacing, 0], [w + self.spacing, h*2 + self.spacing], [w, h*2 + self.spacing]])
        pygame.draw.polygon(self.screen, self.colors['1'], ver_divider)

        # Render value maps and console
        positions = [(1, 1), (0, 1), (1, 0), (0, 0)]
        if self.LIVE_AGENT in self.active_visualizers:
            self._render_map(grid, pos=positions.pop())
        if self.VALUE_FUNCTION in self.active_visualizers and value_map is not None:
            self._render_map(grid, pos=positions.pop(), value_map=value_map)
        if self.DENSITY in self.active_visualizers and density_map is not None:
            self._render_map(grid, pos=positions.pop(), value_map=density_map)
        if self.CONSOLE in self.active_visualizers:
            self._render_console(pos=positions.pop(), message=message)

        pygame.display.flip()
        self.clock.tick(self.fps)

        return True

    def _render_map(self, grid, pos=(0, 0), value_map=None):
        grid_h, grid_w = grid.shape
        w, h = self.size

        offset_left = pos[0] * (w + self.spacing)
        offset_top = pos[1] * (h + self.spacing)

        overshoot_w = w % grid_w
        scale_w = int((w - overshoot_w) / grid_w)
        border_left = math.floor(overshoot_w / 2)
        border_right = overshoot_w - border_left

        overshoot_h = h % grid_h
        scale_h = int((h - overshoot_h) / grid_h)
        border_top = math.floor(overshoot_h / 2)
        border_bottom = overshoot_h - border_top

        # Normalize map
        if value_map is not None:
            if np.sum(value_map) != 0.:
                value_map /= np.max(value_map)
            cmap = plt.get_cmap('hot')
            rgba_img = cmap(value_map) * 255

        for y in range(grid_h):
            for x in range(grid_w):
                cell_type = grid[y, x]
                vertices = np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]])
                vertices[:, 0] = vertices[:, 0] * scale_w + border_left + offset_left
                vertices[:, 1] = vertices[:, 1] * scale_h + border_top + offset_top

                if value_map is not None:
                    color = tuple(rgba_img[y, x])

                    if cell_type == 3 or cell_type == 1:
                        color = self.colors[str(cell_type)]
                else:
                    color = self.colors[str(cell_type)]

                pygame.draw.polygon(self.screen, color, vertices)

        if border_left > 0:
            vertices = np.array([[0 + offset_left, 0 + offset_top], [border_left + offset_left, 0 + offset_top],
                                 [border_left + offset_left, h + offset_top], [0 + offset_left, h + offset_top]])
            pygame.draw.polygon(self.screen, self.colors[str(1)], vertices)

        if border_top > 0:
            vertices = np.array([[0 + offset_left, 0 + offset_top], [w + offset_left, 0 + offset_top],
                                 [w + offset_left, border_top + offset_top], [0 + offset_left, border_top + offset_top]])
            pygame.draw.polygon(self.screen, self.colors[str(1)], vertices)

        if border_right > 0:
            vertices = np.array([[w - border_right + offset_left, 0 + offset_top], [w + offset_left, 0 + offset_top],
                                 [w + offset_left, h + offset_top], [w - border_right + offset_left, h + offset_top]])
            pygame.draw.polygon(self.screen, self.colors[str(1)], vertices)

        if border_bottom > 0:
            vertices = np.array([[0 + offset_left, h - border_bottom + offset_top],
                                 [w + offset_left, h - border_bottom + offset_top], [w + offset_left, h + offset_top],
                                 [0 + offset_left, h + offset_top]])
            pygame.draw.polygon(self.screen, self.colors[str(1)], vertices)

    def _render_console(self, pos, message=None):
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