import math
from collections import deque

import pygame
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt

pygame.font.init()


class MazeVisualizer(object):
    CONSOLE = 0
    LIVE_AGENT = 1
    VALUE_FUNCTION = 2
    DENSITY = 3

    def __init__(self, name="Maze", size=None, fps=60):
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

    def render(self, close=False):
        # Create screen on first call
        if self.screen is None:
            self.screen = pygame.display.set_mode(self.size, 0,32)

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

