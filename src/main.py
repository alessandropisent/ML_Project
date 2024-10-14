from tabular_learning import (
    training,
    select_action_greedy,
    steady_decrease_epsilon,
    test_q,
    select_action_softmax,
    dynamic_base,
)
from utils_functions import show_env
from DQN import Agent
from concat_v import concat_videos_and_delete


import torch
import random
import numpy as np
import gymnasium as gym


def GreedyQlearning():
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    # Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.80  # Discount factor

    num_episodes = 10_000
    max_steps = 1000  # Max steps per episode
    Q = training(
        num_episodes,
        select_action_greedy,
        steady_decrease_epsilon,
        alpha,
        gamma,
        "Epsilon with Greedy policy",
        max_steps,
        env,
    )

    test_q(Q, 10, "Greedy", env)
    concat_videos_and_delete(["Greedy/"])


def dynamicSoftmaxQlearning():
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    # Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.80  # Discount factor

    num_episodes = 10_000
    max_steps = 1000  # Max steps per episode
    Q = training(
        num_episodes,
        select_action_softmax,
        dynamic_base,
        alpha,
        gamma,
        "Base of softmax (Softmax policy)",
        max_steps,
        env,
    )
    test_q(Q, 10, "Softmax", env)
    concat_videos_and_delete(["Softmax/"])


def DQN_learning(need_trainig=True):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # @title Training
    learning_rate = 0.001
    gamma = 0.999
    num_episodes = 50_000
    replay_capacity = 10_000
    batch_size = 128

    agent = Agent(
        learning_rate, gamma, num_episodes, replay_capacity, batch_size, device=device
    )
    if need_trainig:
        agent.train()
        agent.save_csv_metrics("metrics.csv")
        agent.save_model("model.pth")

    else:
        agent.load_model("model.pth")
        agent.load_csv_metrics("metrics.csv")

    agent.plot_metrics()

    ## Testing with visual
    print("CREATING VIDEOS of THE AGENT PLAYING\n\n")
    agent.set_test_size(10)
    agent.test_visual()

    concat_videos_and_delete(["DQN/"])

    ## Testing success rate
    print("Testing the success rate of the agent over 10000 episodes")
    agent.set_test_size(10_000)
    agent.test_relay()
    succ_rate, c_reward, avg_n_steps = agent.cal_mean()
    print("\n")
    print(f"Success Rate: {succ_rate:.3f}")
    print(f"Cumulative Reward avg: {c_reward:.3f}")
    print(f"Average number of steps: {avg_n_steps:.3f}")


if __name__ == "__main__":
    # for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("Greedy Tabular Q-learning")
    GreedyQlearning()

    print("Softmax Tabular Q-learning")
    dynamicSoftmaxQlearning()

    print("Deep Q-learning")
    DQN_learning()
