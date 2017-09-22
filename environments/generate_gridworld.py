import random
from collections import deque

import numpy as np

random.seed()


def fill_cell(gridworld, pos, type=0):
    gridworld[pos[0] - 1: pos[0] + 2, pos[1] - 1: pos[1] + 2] = type


def fill_wall(gridworld, cur_pos, new_pos, type=0):
    pos = np.asarray((cur_pos + new_pos) / 2, dtype=np.int32)
    orientation = np.where(new_pos - cur_pos == 0)[0]
    if orientation == 0:
        gridworld[pos[0] - 1: pos[0] + 2, pos[1]] = type
    elif orientation == 1:
        gridworld[pos[0], pos[1] - 1: pos[1] + 2] = type
    else:
        raise Exception("Something went wrong with the wall, not possible")


def check_free_cell(gridworld, pos, location):
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
    if size is None:
        size = (100, 100)
    if size[0] > 100 or size[1] > 100:
        print("\033[1mMaximum size is (100, 100). Maximum size is used for the generated maze.\033[0m")
    gridworld = np.ones(size, dtype=np.int32)

    start = np.random.randint(1, min(size), size=2)
    fill_cell(gridworld, start, 2)

    stack = deque()
    cur_cell = start
    while True:
        available_cells = []
        for i in range(4):
            cell, available = check_free_cell(gridworld, cur_cell, i)
            if available:
                available_cells.append(cell)
        if not available_cells:
            if not stack:
                break
            cur_cell = stack.pop()
        else:
            new_cell = random.choice(available_cells)
            stack.append(cur_cell)
            fill_wall(gridworld, cur_cell, new_cell)
            fill_cell(gridworld, new_cell)
            cur_cell = new_cell

    possible_goals = np.array(np.where(gridworld == 0))
    goal_idx = np.random.randint(possible_goals.shape[1])
    goal = possible_goals[:, goal_idx]
    fill_cell(gridworld, goal, 3)

    return "Random", gridworld, 1


if __name__ == "__main__":
    from environments.gridworldvisualizer import GridWorldVisualizer
    maze = generate_gridworld()

    render = True
    visualizer = GridWorldVisualizer("RandomGridWorld")
    while render:
        render = visualizer.render(maze)