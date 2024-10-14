from collections import deque
import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils_functions import cumulative, rolling_average
import pandas as pd
import gymnasium as gym


# Replay memory
# We used a double-ended queue because it allowed us to not think about overflowing as the
# library will take care of it, we wrote a sample_D that is used to get the batches
# of the replay memory as if they are already passed throw a Dataloader
class ReplayMemory:
    """
    Replay memory class.

    Parameters:
    - capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.device = device

    def push(
        self,
        state: int,
        action: int,
        next_state: int,
        reward: int,
        done: bool,
        truncated: bool,
    ):

        self.memory.append((state, action, next_state, reward, done, truncated))

    def sample_D(self, batch_size):
        """
        Samples a batch of transitions from the replay memory.
        D stands for Dataloader since it already gives a tuple of:
            (states[batchsize], q_values[batchsize])

        Parameters:
        - batch_size: Number of transitions to sample.

        Returns:
        - A tuple of (states, actions, rewards, dones, next_states). All elements are tensors.
        """

        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones, truncated = zip(*batch)

        return (
            torch.Tensor(states).to(self.device),
            torch.Tensor(actions).to(self.device),
            torch.Tensor(rewards).to(self.device),
            torch.Tensor(dones).to(self.device),
            torch.Tensor(truncated).to(self.device),
            np.array(next_states),
        )

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the replay memory.

        Parameters:
        - batch_size: Number of transitions to sample.

        Returns:
        - A list of transitions (state, action, reward, next_state, done).
        """

        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[idx] for idx in indices]

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]


# NN
class DQN_multiple(nn.Module):
    """
    Deep Q-Network (DQN) class.
    Simple neural network with one hidden layer.

    Parameters:
    - state_dim: Dimension of the state space.
    - action_dim: Dimension of the action space.
    """

    def __init__(self, state_dim, action_dim, device="cpu"):

        super(DQN_multiple, self).__init__()

        self.state_dim = state_dim

        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        ).to(device)
        self.device = device

    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters:
        - x: Input tensor.

        Returns:
        - Output tensor.
        """
        # Move and prepare the states
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        x = x.long()
        # Make them one-hot encoding
        x = torch.nn.functional.one_hot(x, num_classes=self.state_dim)
        x = x.float()
        x = x.to(self.device)

        return self.model(x)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


# Agent: single NN
class Agent:

    def __init__(
        self,
        learning_rate,
        gamma,
        num_episodes,
        replay_capacity,
        batch_size,
        test_size=10,
        device="cpu",
        weight_decay=1e-4,
    ):
        # Creation of Gym env
        self.env = gym.make("Taxi-v3", render_mode="rgb_array")

        self.dqn_model = DQN_multiple(
            self.env.observation_space.n, self.env.action_space.n, device=device
        )

        self.replay_memory = ReplayMemory(replay_capacity, device=device)

        # Hyperparameters
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size

        self.test_size = test_size

        # Simple dictionary that stores the values per each episode
        self.metrics = {
            "loss": [],
            "cumulative_rewards": [],
            "num_steps": [],
            "success_rates": [],
            "value of epsilon": [],
        }

        self.loss = nn.MSELoss()

        self.optimizer = optim.Adam(self.dqn_model.parameters(), lr=learning_rate)

        self.device = device

    def set_test_size(self, test_size):
        self.test_size = test_size

    def greedy_dqn(self, state, episode, max_episodes):
        """
        Selects an action using the epsilon-greedy policy.

        Parameters:
        - state: The current state.
        - episode: The current episode number (0-indexed).
        - max_episodes: The total number of episodes.

        Returns:
        - action: The selected action.
        - epsilon: The current value of epsilon (useful later to graph it).
        """

        EPS_START = 0.999
        EPS_END = 0.001
        EPS_DECAY = max_episodes * 0.7

        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * episode / EPS_DECAY)

        # Explore
        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()

        # Exploit
        else:
            # Here we are using the NN, but we do not want to backpropagate
            # so we specify torch.no_grad
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.int64)
                q_values = self.dqn_model(state)
                action = torch.argmax(q_values).item()

        return action, epsilon

    # Simple function to show the current state of the env
    def im_env(self):
        plt.imshow(self.env.render())
        plt.axis("off")
        plt.show()

    def optimize(self, batch):

        states, actions, rewards, dones, truncates, next_states = batch

        self.optimizer.zero_grad()

        # masrk of the final state, since next_state is always in range(600) but it is -1
        # when it is finished or trunketed == -2
        non_final_mask = [next_states >= 0]
        # The values of all the non final states
        non_final_next_states = torch.Tensor(next_states[next_states >= 0]).to(
            self.device
        )

        # Calculate the explected best action using the nn target
        # We put penalty as default
        q_expected = torch.zeros(self.batch_size).to(self.device)

        # Calculated the estimate for the q table
        q_values = self.dqn_model(states).gather(1, actions.long().unsqueeze(1))

        # Here we do not want to accumulate gradient, we use the NN as a table
        with torch.no_grad():
            q_expected[non_final_mask] = (
                self.dqn_model(non_final_next_states).max(1).values
            )

        # Compute the expected Q values, using gamma update rules and rewards
        q_expected = (q_expected * self.gamma) + rewards

        # loss
        loss = self.loss(q_values, q_expected.unsqueeze(1))

        # backprop
        loss.backward()

        # In-place gradient clipping (to avoid expoding gradient)
        torch.nn.utils.clip_grad_value_(self.dqn_model.parameters(), 100)

        # Adjust learning weights
        self.optimizer.step()

        return loss.item()

    def train(self):
        # training loop
        for episode in tqdm(range(self.num_episodes)):
            state, info = self.env.reset()
            done = False
            truncated = False
            total_reward = 0
            l = 0

            for __ in range(200):
                # Select the action (epsilon is just a nice to have)
                action, epsilon = self.greedy_dqn(state, episode, self.num_episodes)

                # print(f"Episode: {episode}, Action: {action}")

                # Execute the action
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward

                # we decided to represent -1 as the None next_state
                if done or truncated:
                    next_state = -1

                # Store in replay memory
                self.replay_memory.push(
                    state, action, next_state, reward, done, truncated
                )

                if done or truncated:
                    break

                state = next_state

            # if we have enough elements in the memory we train the nn
            if len(self.replay_memory) >= self.batch_size:
                batch = self.replay_memory.sample_D(self.batch_size)
                l = self.optimize(batch)

            # print(f"Episode: {episode}, Total reward: {total_reward}", end="\r")

            # Append Metrics
            self.metrics["cumulative_rewards"].append(total_reward)
            if done:
                self.metrics["success_rates"].append(1)
            else:
                self.metrics["success_rates"].append(0)

            self.metrics["num_steps"].append(__)
            self.metrics["loss"].append(l)
            self.metrics["value of epsilon"].append(epsilon)

    def plot_metrics(self):
        # 5 different subplots for the cumulative reward, success rate and number of steps
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 15))

        # Plot the Loss
        ax1.plot(self.metrics["loss"], label="loss")
        ax1.plot(
            cumulative(self.metrics["loss"], 100),
            label="cumulative loss over 100 episodes",
        )
        ax1.legend()
        ax1.set_title("Loss")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Loss")

        # Plot the cumulative reward
        ax2.plot(
            self.metrics["cumulative_rewards"], label="cumulative reward per episode"
        )
        ax2.plot(
            rolling_average(self.metrics["cumulative_rewards"], 100),
            label="reward moving average over 100 episodes",
        )
        ax2.legend()
        ax2.set_title("Cumulative Reward")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Reward")

        # Plot the success rate
        ax3.plot(self.metrics["success_rates"], label="success rate per episode")
        ax3.plot(
            rolling_average(self.metrics["success_rates"], 100),
            label="success rate moving average over 100 episodes",
        )
        ax3.legend()
        ax3.set_title("Success Rate")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Success Rate")

        # Plot the number of steps
        ax4.plot(self.metrics["num_steps"], label="number of steps taken per episode")
        ax4.plot(
            rolling_average(self.metrics["num_steps"], 100),
            label="number of steps moving average over 100 episodes",
        )
        ax4.legend()
        ax4.set_title("Number of Steps")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Number of Steps")

        # Plot the Value of Epsilon
        ax5.plot(self.metrics["value of epsilon"])
        ax5.set_title("Value of epsilon")
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Value of epsilon")

        plt.tight_layout()
        plt.show()

    def save_csv_metrics(self, path):
        df = pd.DataFrame(self.metrics)
        df.to_csv(path)

    def load_csv_metrics(self, path):
        self.metrics = pd.read_csv(path)

    def save_model(self, path):
        torch.save(self.dqn_model.state_dict(), path)

    def load_model(self, path):
        self.dqn_model.load_state_dict(torch.load(path))

    def return_metrics(self):
        return self.metrics

    # This function is to visually test the model
    # it will print the varius fram of the Agent playing the game
    def test_visual(self):
        env_video = gym.wrappers.RecordVideo(
            self.env, video_folder="video_DQN", episode_trigger=lambda x: True
        )
        for episode in range(self.test_size):
            state, _ = env_video.reset(seed=episode)
            done = False
            truncated = False
            total_reward = 0
            env_video.render()

            while not (done or truncated):
                action = torch.argmax(self.dqn_model(state)).item()
                next_state, reward, done, truncated, _ = env_video.step(action)
                total_reward += reward
                state = next_state

                env_video.render()

            print(f"Episode: {episode}, Total reward: {total_reward}")

    # this function is to test how reliable is the model
    # it will just get the varius metrics just using the Policy network
    def test_relay(self):

        self.metrics = {
            "loss": [],
            "cumulative_rewards": [],
            "num_steps": [],
            "success_rates": [],
            "value of epsilon": [],
        }

        with torch.no_grad():
            for episode in tqdm(range(self.test_size)):
                state, _ = self.env.reset()
                done = False
                truncated = False
                total_reward = 0

                for __ in range(500):

                    action = torch.argmax(self.dqn_model(state)).item()
                    next_state, reward, done, truncated, _ = self.env.step(action)
                    total_reward += reward
                    state = next_state

                    if done or truncated:
                        break

                self.metrics["cumulative_rewards"].append(total_reward)
                if done:
                    self.metrics["success_rates"].append(1)
                else:
                    self.metrics["success_rates"].append(0)

                self.metrics["num_steps"].append(__)
                self.metrics["loss"].append(0)
                self.metrics["value of epsilon"].append(0)

    # Simple function that get the mean
    def cal_mean(self):
        """
        Returns:
        - A tuple of (success_rate, cumulative_reward, num_steps)
        """
        return (
            np.mean(self.metrics["success_rates"]),
            np.mean(self.metrics["cumulative_rewards"]),
            np.mean(self.metrics["num_steps"]),
        )
