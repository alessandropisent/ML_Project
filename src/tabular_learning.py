import gymnasium as gym
import torch
import numpy as np
import math
from utils_functions import plot_metrics
from tqdm import tqdm

#env = gym.make('Taxi-v3',render_mode="rgb_array")
#env.time_limit = 200

# Policy Greedy
def select_action_greedy(Q, state, episode, max_episodes, update_epsilon, env):
    """
    Selects an action using the epsilon-greedy policy.

    Parameters:
    - Q: The Q-table.
    - state: The current state.
    - episode: The current episode number (0-indexed).
    - max_episodes: The total number of episodes.
    - update_epsilon: A function that updates the epsilon value.

    Returns:
    - action: The selected action.
    """

    epsilon = update_epsilon(episode, max_episodes)

    if torch.rand(1).item() < epsilon:  # Exploration
        return env.action_space.sample()
    else:  # Exploitation
        return torch.argmax(Q[state]).item()

# Decrease of epsilon greedy
def steady_decrease_epsilon(episode, max_episodes):
    return 1.0 - episode / max_episodes

# Softmax policy
def select_action_softmax(Q, state, episode, max_episodes, update_base, env):
    """
    Selects an action using the softmax policy.

    Parameters:
    - Q: The Q-table.
    - state: The current state.
    - episode: The current episode number (0-indexed).
    - max_episodes: The total number of episodes.
    - update_base: A function that updates the base value.

    Returns:
    - action: The selected action.

    """

    #probs = torch.nn.functional.softmax(Q[state], dim=0)

    base = update_base(episode, max_episodes)


    pow = torch.pow(base, Q[state])
    probs = pow / torch.sum(pow)

    return torch.multinomial(probs, 1).item()

# Esponential decrease
def dynamic_base(episode, max_episodes, initial_base=20, min_base=math.e, decay_rate=0.20):
    """
    Returns the base (or temperature) for the softmax policy depending on the current episode.

    Parameters:
    - episode: The current episode number (0-indexed).
    - max_episodes: The total number of episodes.
    - initial_base: The initial base (or temperature) at the start of training.
    - min_base: The minimum base (or temperature) to avoid it going too low.
    - decay_rate: The rate at which the base decays.

    Returns:
    - base: The base value for the softmax policy.
    """
    # Calculate the decayed base value
    base = initial_base * (decay_rate ** (episode / max_episodes))

    # Ensure the base doesn't fall below the minimum allowed value
    return max(base, min_base)

# Rolling avg function
def rolling_average(data, window_size):
    """
    Computes the rolling average

    Parameters:
    - data: List of data points.
    - window_size: The size of the rolling window.

    Returns:
    - A list of rolling averages.
    """

    # Calculate the rolling averages
    rolling_averages = np.convolve(data, np.ones(window_size)/ window_size, 'valid')

    return rolling_averages.tolist()

# Training function for Tabular Q-learning
def training(num_episodes, F_policy, F_update_p, 
             alpha, gamma, string_param, 
             max_steps, env):
    """
    Trains the Q-learning agent.

    Parameters:
    - num_episodes: max number of episodes for the training
    - F_policy: function that selects the policy and the action.
    - F_update_p: function that updates the parameters of the policy.
    - alpha: Hyperparameter for the learning rate.
    - gamma: Hyperparameter for the discount factor.
    - early_stopping: Boolean flag indicating whether early stopping is enabled.

    Returns:
    - cumulative_rewards: List of cumulative rewards for each episode.
    - num_steps: List of number of steps taken for each episode.
    - success_rates: List of success rates for each episode. [0 if not successful,1 otherwise]
    """

    # Initialize Q-table (state-action value table)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = torch.zeros(n_states, n_actions)

    print("The Q-table is in: ", Q.device)

    # Metrics
    cumulative_rewards = []
    success_rates = []
    num_steps = []

    print(f"Training for: {num_episodes:} episodes")
    # for loop for the episodes
    for episode in tqdm(range(num_episodes)):

        # Reset the environment and get the initial state
        state, _ = env.reset()

        # Reset the cumulative reward for the episode
        temp_cum_reward = 0

        for step in range(max_steps):

            # choose the action according the function it was given and update
            # the parameter for the policy
            action = F_policy(Q, state, episode, num_episodes, F_update_p, env)

            # Execute the action
            next_state, reward, done, truncated, _ = env.step(action)

            # update Q
            Q[state,action] = (1-alpha)*Q[state,action] + alpha*(reward + gamma*torch.max(Q[next_state]))

            # next state
            state = next_state

            # metrics - cumulative reward
            temp_cum_reward += reward

            # stop if we are done or we are over the limit
            if done or truncated:
                break

        # metrics - append data of the episode
        cumulative_rewards.append(temp_cum_reward)
        if done:
            success_rates.append(1)
        else:
            success_rates.append(0)
        num_steps.append(step)


    # Plot the metrics
    plot_metrics(cumulative_rewards, num_steps, success_rates,
     [F_update_p(i, max_steps) for i in range(max_steps)], string_param)

    return Q

# Test function
def test_q(Q, num_episodes,nameAgent, env):
    env_video = gym.wrappers.RecordVideo(env, video_folder='video_'+nameAgent, episode_trigger=lambda x: True)
    for episode in range(num_episodes):
        state, _ = env_video.reset()
        done = False
        truncated = False
        total_reward = 0
        env_video.render()

        while not (done or truncated):
            action = torch.argmax(Q[state]).item()
            next_state, reward, done, truncated, _ = env_video.step(action)
            total_reward += reward
            state = next_state

            env_video.render()

        print(f"Episode: {episode}, Total reward: {total_reward}")