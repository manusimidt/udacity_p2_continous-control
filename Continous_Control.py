from unityagents import UnityEnvironment
import numpy as np


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
        action = agent.act(state, epsilon=0.05)
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break
    print(f"Agent achieved a score of {score}")


if __name__ == '__main__':
    env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)

    env.close()
