"""Module for generating random GridWorld

Defines methods necessary for generating a random gridworld. The generated
gridworld is a numpy array with dtype int where each entry is a state.

Attributes in the gridworld are defined with the following types:
    0   Empty
    1   Wall
    2   Start state
    3   Goal state
"""
import random
from collections import deque

import numpy as np

random.seed()


def _fill_cell(gridworld, pos, type=0):
    """Fill cell with type

    Cell at position pos and surrounding it is filled with type. Resulting in a
    3x3 grid with center pos.

    Args:
        gridworld:  numpy array of the gridworld
        pos:        2D position to be filled
        type:       type to fill with

    Returns:

    """
    gridworld[pos[0] - 1: pos[0] + 2, pos[1] - 1: pos[1] + 2] = type


def _fill_wall(gridworld, cur_pos, new_pos, type=0):
    """

    Args:
        gridworld:
        cur_pos:
        new_pos:
        type:

    Returns:

    """
    pos = np.asarray((cur_pos + new_pos) / 2, dtype=np.int32)
    orientation = np.where(new_pos - cur_pos == 0)[0]
    if orientation == 0:
        gridworld[pos[0] - 1: pos[0] + 2, pos[1]] = type
    elif orientation == 1:
        gridworld[pos[0], pos[1] - 1: pos[1] + 2] = type
    else:
        raise Exception("Something went wrong with the wall, not possible")


def _check_free_cell(gridworld, pos, location):
    """

    Args:
        gridworld:
        pos:
        location:

    Returns:

    """
    pos = pos.copy()
    if location == 0:
        pos += np.array([4, 0])
    elif location == 1:
        pos += np.array([0, -4])
    elif location == 2:
        pos += np.array([-4, 0])
    elif location == 3:
        pos += np.array([0, 4])
    else:
        raise Exception("Choose from available locations: [0, 1, 2, 3]")
    free = False
    if np.sum(gridworld[pos[0] - 1: pos[0] + 2, pos[1] - 1: pos[1] + 2]) == 9:
        free = True
    return pos, free


def generate_gridworld(size=None):
    """

    Args:
        size:

    Returns:

    """
    if size is None:
        size = (100, 100)
    if size[0] > 100 or size[1] > 100:
        print("\033[1mMaximum size is (100, 100). Maximum size is used for the generated maze.\033[0m")
    gridworld = np.ones(size, dtype=np.int32)

    start = np.random.randint(1, min(size), size=2)
    _fill_cell(gridworld, start, 2)

    stack = deque()
    cur_cell = start
    while True:
        available_cells = []
        for i in range(4):
            cell, available = _check_free_cell(gridworld, cur_cell, i)
            if available:
                available_cells.append(cell)
        if not available_cells:
            if not stack:
                break
            cur_cell = stack.pop()
        else:
            new_cell = random.choice(available_cells)
            stack.append(cur_cell)
            _fill_wall(gridworld, cur_cell, new_cell)
            _fill_cell(gridworld, new_cell)
            cur_cell = new_cell

    possible_goals = np.array(np.where(gridworld == 0))
    goal_idx = np.random.randint(possible_goals.shape[1])
    goal = possible_goals[:, goal_idx]
    _fill_cell(gridworld, goal, 3)

    return "Random", gridworld, 1


if __name__ == "__main__":
    from smartexploration.environments.gridworldvisualizer import GridWorldVisualizer
    maze = generate_gridworld()

    render = True
    visualizer = GridWorldVisualizer("RandomGridWorld")
    while render:
        render = visualizer.render(maze)