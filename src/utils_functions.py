import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(cumulative_rewards, num_steps, success_rates, param, string_param):
    """
    Function that plots the metrics.

    Parameters:
    - cumulative_rewards: List of cumulative rewards for each episode.
    - num_steps: List of number of steps taken for each episode.
    - success_rates: List of success rates for each episode. [0 if not succesful,1 otherwhise]
    - param: List of parameter values for each episode.
    - string_param: String representation of the parameter, used in the plot of the parameter
                    variation throuout the training.

    Returns:
    - None (plots the metrics with matplotlib).
    """

    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))

    # Cumulative rewards
    axs[0].plot(cumulative_rewards, label="actual data")
    axs[0].plot(rolling_average(cumulative_rewards, 100), label="rolling avg 100")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Cumulative Reward")
    axs[0].set_title("Cumulative Reward per Episode")
    axs[0].legend()

    # number of steps taken by the agent per episode
    axs[1].plot(num_steps, label="actual data")
    axs[1].plot(rolling_average(num_steps, 100), label="rolling avg 100")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Number of Steps")
    axs[1].set_title("Number of Steps per Episode")
    axs[1].legend()

    # Rolling Avg of success per episonde
    axs[2].plot(rolling_average(success_rates, 100))
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Rolling Avg of Success Rate")
    axs[2].set_title("Rolling Avg of Success Rate")

    # Value of the paramenter per episode
    axs[3].plot(param)
    axs[3].set_xlabel("Episode")
    axs[3].set_ylabel("Parameter")
    axs[3].set_title(f"Value of {string_param} per Episode")

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

#@title Rolling avg function
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

#@title cumulative function
def cumulative(data, window_size):
    """
    Computes the cumulative sum

    Parameters:
    - data: List of data points.
    - window_size: The size of the rolling window.

    Returns:
    - A list of cumulative sums.
    """


    # Use a rolling window sum function
    result = np.convolve(data, np.ones(window_size), 'valid')

    return result.tolist()

def show_env(env):
    plt.imshow(env.render())
    plt.axis('off')
    plt.show()