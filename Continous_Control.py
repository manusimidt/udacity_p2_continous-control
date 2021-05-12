from collections import deque

import numpy as np
import pandas as pd
import torch
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt

from Agent import Agent


def watch_agent_from_pth_file(env: UnityEnvironment, brain_name: str, agent: Agent, file_path_actor: str,
                              file_path_critic: str) -> None:
    agent.actor_local.load_state_dict(torch.load(file_path_actor))
    agent.critic_local.load_state_dict(torch.load(file_path_critic))
    agent.actor_local.eval()
    agent.critic_local.eval()
    watch_agent(env, brain_name, agent)


def watch_agent(env: UnityEnvironment, brain_name: str, agent: Agent) -> None:
    """
    Shows agent simulation
    :param env:
    :param brain_name:
    :param agent:
    :return:
    """
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score

    while True:
        action = agent.act(state)
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break
    print(f"Agent achieved a score of {score}")


def train_agent(env: UnityEnvironment, brain_name: str, agent: Agent, n_episodes: int, max_steps: int = 1500) -> []:
    """
    Trans the agent for n episodes
    :param env:
    :param brain_name:
    :param agent:
    :param n_episodes: number of episodes to train
    :param max_steps: max amount of steps
    :return: returns an array containing the score of every episode
    """
    scores: [int] = []
    # store the last 100 scores into a queue to check if the agent reached the goal
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        state = env_info.vector_observations[0]
        score = 0

        # the environment will end the episode after n steps, thus no manual termination of the episode is needed
        for a in range(max_steps):
            action: int = agent.act(state, add_noise=False)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward

            agent.step((state, action, reward, next_state, done))

            state = next_state
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        if i_episode % 5 == 0:
            print(f"""Episode {i_episode}: Average Score: {np.mean(scores_window):.2f}""")

        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint-actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint-critic.pth')
            break
    return scores


def plot_scores(scores: [int], sma_window: int = 50) -> None:
    """
    Plots a line plot of the scores.
    The function expects the score of the first episode at scores[0] and the last episode at scores[-1]
    :param scores:
    :param sma_window: Simple Moving Average rolling window
    :return:
    """
    # calculate moving average of the scores
    series: pd.Series = pd.Series(scores)
    window = series.rolling(window=sma_window)
    scores_sma: pd.Series = window.mean()

    # plot the scores
    fig = plt.figure(figsize=(12, 5))

    plot1 = fig.add_subplot(121)
    plot1.plot(np.arange(len(scores)), scores)
    plot1.set_ylabel('Score')
    plot1.set_xlabel('Episode #')
    plot1.set_title("Raw scores")

    plot2 = fig.add_subplot(122)
    plot2.plot(np.arange(len(scores_sma)), scores_sma)
    plot2.set_ylabel('Score')
    plot2.set_xlabel('Episode #')
    plot2.set_title(f"Moving Average(window={sma_window})")

    plt.show()


if __name__ == '__main__':
    # _env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')
    print("Loading Environment...")
    _env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')
    print("Environment loaded")

    # get the default brain
    _brain_name = _env.brain_names[0]
    _brain = _env.brains[_brain_name]

    _action_size: int = 4
    _state_size: int = 33

    _agent = Agent(_state_size, _action_size,
                   gamma=0.99, lr_actor=0.0002, lr_critic=0.0003, tau=0.002, weight_decay=0.0001,
                   buffer_size=1000000, batch_size=128, seed=0)

    # with this boolean you can decide if you just want to watch an agent or train the agent yourself
    watch_only = False
    if watch_only:
        watch_agent_from_pth_file(_env, _brain_name, _agent, './checkpoint-actor.pth', './checkpoint-critic.pth')
    else:
        scores = train_agent(_env, _brain_name, _agent, n_episodes=500, max_steps=1500)
        watch_agent(_env, _brain_name, _agent)
        plot_scores(scores=scores, sma_window=10)

    _env.close()
