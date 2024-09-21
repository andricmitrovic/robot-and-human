import os

import gymnasium as gym
import matplotlib.pyplot as plt
import register_env  # Make sure to import your registration module
import numpy as np

np.random.seed(42)


def random_agent(env, verbose = 0):
    EPISODES = 2000
    STEPS = 20

    times = []
    stress = []
    # Use the environment
    for i_episode in range(EPISODES):
        observation, _ = env.reset()
        stress_peak = 0
        if verbose:
            env.render()
        for t in range(STEPS):
            remainingTasks = observation[3]
            action = env.action_space.sample(remainingTasks)
            observation, reward, terminated, truncated, info = env.step(action)
            stress_peak = min(stress_peak, reward)
            if verbose:
                print(f"Action: {action}\nNew State: {observation}\nReward(Stress): {reward}\n")
            if terminated:
                if verbose:
                    print(f"Episode finished after {t+1} timesteps")
                times.append(observation[0].item())
                break
        stress.append(stress_peak)
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
    stress = np.array(stress)
    running_mean_stress = np.cumsum(stress) / (np.arange(1, len(stress) + 1))

    # Smooth the running mean using a moving average
    window_size = 50  # You can adjust the window size for more or less smoothing

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    smoothed_running_mean = moving_average(running_mean_stress, window_size)

    # Plot smoothed running mean of stress
    plt.plot(range(len(smoothed_running_mean)), smoothed_running_mean, color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Running Mean Stress')
    plt.title('Smoothed Running Mean of Stress over Episodes')
    plt.savefig(f'{dir_path}/smoothed_running_mean_stress.png')
    plt.close()


if __name__ == '__main__':
    env = gym.make('CollaborationEnv-v0')
    random_agent(env, verbose = 0)
    env.close()
