import os

import gymnasium as gym
import matplotlib.pyplot as plt
import register_env  # Make sure to import your registration module
import numpy as np
from tqdm import tqdm

np.random.seed(42)


def random_agent(env, verbose = 0):
    EPISODES = 2000
    STEPS = 100

    times = []
    rewards = []
    # Use the environment
    for i_episode in tqdm(range(EPISODES)):
        observation, _ = env.reset()
        if verbose:
            env.render()
        for t in range(STEPS):
            remainingTasks = observation[3]
            action = env.action_space.sample(remainingTasks)
            observation, reward, terminated, truncated, info = env.step(action)
            if verbose:
                print(f"Action: {action}\nNew State: {observation}\nReward(Stress): {reward}\n")
            if terminated:
                if verbose:
                    print(f"Episode finished after {t+1} timesteps")
                times.append(observation[0].item())
                rewards.append(reward)
                break
    times = np.array(times)
    mean_exec_time = np.mean(times)
    std_exec_time = np.std(times)
    print(f"Mean exec time: {mean_exec_time}")
    print(f"Standard Deviation exec time: {std_exec_time}")

    dir_path = "../output/random_agent/"
    os.makedirs(dir_path, exist_ok=True)

    plt.hist(times, bins=50, color='orange', edgecolor='black')
    plt.xlabel('Total exec time')
    plt.ylabel('Frequency')
    # plt.title(f'Histogram of Total Execution Times')
    plt.savefig(f'{dir_path}/exec_time_hist.png')
    plt.close()

    # Calculate running mean of stress
    rewards = np.array(rewards)
    mean_rewards = np.mean(rewards)
    print(f"Mean reward: {mean_rewards}")
    plt.scatter(range(len(rewards)), rewards, color='blue', label='Reward', s=2)
    plt.axhline(mean_rewards, color='red', label='Mean reward')
    plt.title('Episodic reward')
    plt.legend()
    plt.savefig(f'{dir_path}/rewards.png')
    plt.close()

    # running_mean_stress = np.cumsum(stress) / (np.arange(1, len(stress) + 1))
    #
    # # Smooth the running mean using a moving average
    # window_size = 50  # You can adjust the window size for more or less smoothing
    #
    # def moving_average(data, window_size):
    #     return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    #
    # smoothed_running_mean = moving_average(running_mean_stress, window_size)

    # # Plot smoothed running mean of stress
    # plt.figure(figsize=(12, 6))
    # plt.scatter(range(len(stress)), stress, alpha=0.8, color='lightgray', label='Raw Stress Data', s=5)
    # plt.plot(range(len(smoothed_running_mean)), smoothed_running_mean, color='blue', label='Smoothed Running Mean')
    # plt.xlabel('Episode')
    # plt.ylabel('Stress')
    # plt.title('Smoothed Running Mean of Stress over Episodes')
    # plt.axhline(y=np.mean(stress), color='red', linestyle='--', label='Baseline Mean')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'{dir_path}/smoothed_running_mean_stress.png')
    # plt.close()


if __name__ == '__main__':
    env = gym.make('CollaborationEnv-v0', operator='stress+')
    random_agent(env, verbose = 0)
    env.close()
