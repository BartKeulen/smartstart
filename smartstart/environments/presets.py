"""Preset GridWorlds module

Are used by the :obj:`~smartexploration.environments.gridworld.GridWorld` to
generate the gridworld.
"""


def easy():
    """Easy GridWorld

    Returns
    -------
    :obj:`str`
        name
    double :obj:`list` of `int`
        layout
    :obj:`int`
        scale
    """
    name = "Easy"
    layout = [[3, 0, 0],
              [1, 1, 0],
              [1, 1, 0],
              [2, 0, 0]]
    return name, layout


def medium():
    """Medium GridWorld

    Returns
    -------
    :obj:`str`
        name
    double :obj:`list` of `int`
        layout
    :obj:`int`
        scale
    """
    name = "Medium"
    layout = [[3, 0, 0, 0],
              [1, 1, 1, 0],
              [0, 0, 0, 0],
              [0, 1, 1, 1],
              [0, 0, 0, 2]]
    return name, layout


def hard():
    """Hard GridWorld

    Returns
    -------
    :obj:`str`
        name
    double :obj:`list` of `int`
        layout
    :obj:`int`
        scale
    """
    name = "Hard"
    layout = [[3, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 0],
              [2, 0, 0, 0, 0, 0, 0]]
    return name, layout


def maze():
    name = "Maze"
    layout = [[0, 0, 0, 0, 0, 0, 0, 1, 3],
              [0, 0, 1, 0, 0, 0, 0, 1, 0],
              [2, 0, 1, 0, 0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    return name, layout


def maze_large():
    name = "Maze"
    layout = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
              [0, 2, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    return name, layout


def misleading():
    name = "Misleading"
    layout = [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 4, 0, 0, 1, 1, 1, 1, 1, 0, 0, 3, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 2, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
    return name, layout


def extreme():
    """Extreme GridWorld

    Returns
    -------
    :obj:`str`
        name
    double :obj:`list` of `int`
        layout
    :obj:`int`
        scale
    """
    name = "Extreme"
    layout = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3],
              [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
              [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
              [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0],
              [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
              [1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
              [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
              [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
              [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
              [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
              [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
              [0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
              [1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
              [2, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]]
    return name, layout
