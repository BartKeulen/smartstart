import random

from smartstart.utilities.datacontainers import Episode
from smartstart.environments.gridworld import GridWorld


def run_test(summary, step=1, render=False):
    if summary.env_name == 'GridWorldEasy':
        env = GridWorld.generate(GridWorld.EASY)
    elif summary.env_name == 'GridWorldMedium':
        env = GridWorld.generate(GridWorld.MEDIUM)
    elif summary.env_name == 'GridWordHard':
        env = GridWorld.generate(GridWorld.HARD)
    elif summary.env_name == 'GridWorldExtreme':
        env = GridWorld.generate(GridWorld.EXTREME)
    else:
        raise NotImplementedError('%s environment is not implemented yet' % summary.env_name)

    for iter, episode in enumerate(summary.episodes):
        if iter % step == 0:
            test_episode = Episode(episode.iter)
            test_episode.set_value_function(episode.value_function)
            play_episode(env, summary.max_steps, test_episode, render)
            summary.append_test(test_episode)

    return summary


def play_episode(env, max_steps, test_episode, render):
    obs = env.reset()

    for i in range(max_steps):
        if render:
            env.render()

        action = get_action(env, test_episode.value_function, obs)

        obs, reward, done, _ = env.step(action)

        test_episode.append(reward)

        if done:
            break

    if render:
        env.render()
    print("Episode: %d, total reward: %.2f" % (test_episode.iter, test_episode.reward))


def get_action(env, value_function, obs):
    actions = env.possible_actions(obs)
    max_value = -float('inf')
    max_actions = []

    for action in actions:
        idx = tuple(obs) + (action,)
        value = value_function[idx]

        if value > max_value:
            max_value = value
            max_actions = [action]
        elif value == max_value:
            max_actions.append(action)

    return random.choice(max_actions)