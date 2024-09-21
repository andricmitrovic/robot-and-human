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
                break
    times = np.array(times)
    mean_exec_time = np.mean(times)
    std_exec_time = np.std(times)
    print(f"Mean exec time: {mean_exec_time}")
    print(f"Standard Deviation exec time: {std_exec_time}")

    dir_path = "./output/random_agent/"
    os.makedirs(dir_path, exist_ok=True)
    plt.hist(times, bins=50, color='orange', edgecolor='black')
    plt.savefig(f'{dir_path}/exec_time_hist.png')
    plt.show()


if __name__ == '__main__':
    env = gym.make('CollaborationEnv-v0')
    random_agent(env, verbose = 0)
    env.close()
