from collections import deque
import random

import numpy as np

random.seed()


def fill_cell(maze, pos, type=0):
    maze[pos[0] - 1: pos[0] + 2, pos[1] - 1: pos[1] + 2] = type


def fill_wall(maze, cur_pos, new_pos, type=0):
    pos = np.asarray((cur_pos + new_pos) / 2, dtype=np.int32)
    orientation = np.where(new_pos - cur_pos == 0)[0]
    if orientation == 0:
        maze[pos[0] - 1: pos[0] + 2, pos[1]] = type
    elif orientation == 1:
        maze[pos[0], pos[1] - 1: pos[1] + 2] = type
    else:
        raise Exception("Something went wrong with the wall, not possible")


def check_free_cell(maze, pos, location):
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
    if np.sum(maze[pos[0] - 1: pos[0] + 2, pos[1] - 1: pos[1] + 2]) == 9:
        free = True
    return pos, free


def generate_maze(size=None):
    if size is None:
        size = (100, 100)
    if size[0] > 100 or size[1] > 100:
        print("\033[1mMaximum size is (100, 100). Maximum size is used for the generated maze.\033[0m")
    maze = np.ones(size, dtype=np.int32)

    start = np.random.randint(1, min(size), size=2)
    fill_cell(maze, start, 2)

    stack = deque()
    cur_cell = start
    while True:
        available_cells = []
        for i in range(4):
            cell, available = check_free_cell(maze, cur_cell, i)
            if available:
                available_cells.append(cell)
        if not available_cells:
            if not stack:
                break
            cur_cell = stack.pop()
        else:
            new_cell = random.choice(available_cells)
            stack.append(cur_cell)
            fill_wall(maze, cur_cell, new_cell)
            fill_cell(maze, new_cell)
            cur_cell = new_cell

    possible_goals = np.array(np.where(maze == 0))
    goal_idx = np.random.randint(possible_goals.shape[1])
    goal = possible_goals[:, goal_idx]
    fill_cell(maze, goal, 3)

    return "Random", maze, 1


if __name__ == "__main__":
    from environments.gridworldvisualizer import GridWorldVisualizer
    maze = generate_maze()

    render = True
    visualizer = GridWorldVisualizer("RandomMaze")
    while render:
        render = visualizer.render(maze)