import logging

import numpy as np
from smartstart.agents.smartstart import SmartStart

from smartstart.utilities.utilities import Summary, Episode
from smartstart.utilities.utilities import compare_policies

logger = logging.getLogger(__name__)

RENDER = False
RENDER_TEST = False
RENDER_EPISODE = False

SEED = None
MAX_STEPS = 50000
MAX_STEPS_EPISODE = 100
TEST_FREQ = 0


def train(env, agent, true_state_action_values=None):
    global RENDER, RENDER_EPISODE
    np.random.seed(SEED)

    summary = Summary(env, agent)

    i_episode = 0
    total_steps = 0
    test_count = TEST_FREQ
    while total_steps < MAX_STEPS:
        episode = Episode(i_episode)
        obs = env.reset()

        reached_terminal_state = False
        reached_smart_start = False
        smart_start_state = None
        if agent.__class__ == SmartStart and i_episode > 0 and np.random.rand() <= agent.eta:
            smart_start_state = agent.get_smart_start_state()

            logger.debug('[SS] - Smart Start State: [%d, %d]' % smart_start_state)

            agent.fit_model_and_optimize(smart_start_state)

            for i in range(MAX_STEPS_EPISODE):
                action = agent.get_action(obs, use_ss_policy=True)
                obs_tp1, reward, done = env.step(action)

                if RENDER:
                    RENDER = render(env, agent)

                agent.update(obs, action, reward, obs_tp1, done)

                episode.add(reward)
                total_steps += 1
                test_count += 1

                obs = obs_tp1

                if np.array_equal(obs, smart_start_state):
                    logger.debug('[SS] - Reached smart start state: [%d, %d] == [%d, %d]' % (smart_start_state + tuple(obs)))
                    reached_smart_start = True
                    break

                if done:
                    logger.debug('[SS] Reach terminal state before smart start state')
                    reached_terminal_state = True
                    break

        if smart_start_state is not None and not reached_smart_start and not reached_terminal_state:
            logger.debug('[SS] - Did not reach smart start state: [%d, %d] != [%d, %d]' % (smart_start_state + tuple(obs)))

        if not reached_terminal_state:
            for _ in range(MAX_STEPS_EPISODE - episode.steps):
                action = agent.get_action(obs, use_ss_policy=False)
                obs_tp1, reward, done = env.step(action)

                if RENDER:
                    RENDER = render(env, agent)

                agent.update(obs, action, reward, obs_tp1, done)

                episode.add(reward)
                total_steps += 1
                test_count += 1

                obs = obs_tp1

                if done:
                    break

        summary.add_train_episode(episode)
        logger.info('[TRAIN] - Episode: %d, steps: %d, reward %.2f' % (i_episode, episode.steps, episode.reward))

        if RENDER_EPISODE:
            RENDER_EPISODE = render(env, agent)

        if TEST_FREQ != 0 and (i_episode == 0 or test_count >= TEST_FREQ or total_steps >= MAX_STEPS):
            test_episode = test(env, agent, i_episode)
            summary.add_test_episode(test_episode)
            logger.info(
                '[TEST]  - Episode: %d, steps: %d, reward %.2f' % (i_episode, test_episode.steps, test_episode.reward))

            if true_state_action_values is not None:
                policy_percentage_correct, count, tot_size = compare_policies(true_state_action_values, agent.get_state_action_values(), env)
                summary.add_test_policy(i_episode, policy_percentage_correct)
                logger.info('[TEST]  - Correct policy: %.2f (%d / %d)' % (policy_percentage_correct, count, tot_size))

            test_count = 0

        i_episode += 1

    if RENDER or RENDER_EPISODE:
        env.render(close=True)

    return summary


def test(env, agent, episode=0):
    global RENDER_TEST
    episode = Episode(episode)
    obs = env.reset()

    render_test = RENDER_TEST
    for _ in range(MAX_STEPS_EPISODE):
        action = agent.get_greedy_action(obs)
        obs_tp1, reward, done = env.step(action)
        if render_test:
            render_test = render(env, agent)
        episode.add(reward)
        obs = obs_tp1

        if done:
            break

    if render_test:
        env.render(close=True)

    return episode


def render(env, agent, message=None):
    density_map = None
    if hasattr(agent, 'counter'):
        density_map = agent.counter.get_state_visitation_counts()
    value_map = agent.get_state_values()
    return env.render(value_map=value_map, density_map=density_map, message=message)
