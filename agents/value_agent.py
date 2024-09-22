import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import register_env  # Make sure to import your registration module
from tqdm import tqdm

np.random.seed(42)


# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)  # Combined input
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Output is a single value

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output the value


class ValueAgent:
    def __init__(self, env, SAMPLE_SIZE):
        self.env = env

        self.state_dim = 23  # 3 state components + 20 remaining tasks
        self.action_dim = 13 * 2  # 13 tasks for human and 13 tasks for robot

        self.SAMPLE_SIZE = SAMPLE_SIZE # Number of actions to sample

        self.value_network = ValueNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)

    def select_best_action(self, state, epsilon):
        valid_actions = []
        remaining_tasks = state[3]

        # Exploration
        if np.random.uniform() < epsilon:
            return self.env.action_space.sample(remaining_tasks)

        # Sample valid actions
        for _ in range(self.SAMPLE_SIZE):
            action = self.env.action_space.sample(remaining_tasks)
            valid_actions.append(action)

        # Evaluate each action using the Value Network
        action_values = []

        for action in valid_actions:
            # Prepare state and action tensors
            state_tensor = torch.FloatTensor(np.concatenate(([state[0], state[1], state[2]], state[3]))).unsqueeze(0)
            action_tensor = torch.FloatTensor(self.encode_action(action)).unsqueeze(0)
            value = self.value_network(state_tensor, action_tensor).item()
            action_values.append(value)

        # Select the best action based on the highest value
        best_action_index = np.argmax(action_values)
        best_action = valid_actions[best_action_index]

        return best_action

    def encode_action(self, action):
        encoding_human = {1:0,
                          2:1,
                          3:2,
                          4:3,
                          5:4,
                          6:5,
                          7:6,
                          8:7,
                          9:8,
                          10:9,
                          12:10,
                          13:11,
                          14:12}
        encoding_robot = {20: 0,
                          19: 1,
                          18: 2,
                          17: 3,
                          16: 4,
                          15: 5,
                          14: 6,
                          13: 7,
                          12: 8,
                          11: 9,
                          9: 10,
                          8: 11,
                          7: 12}
        # Encode action for human and robot separately
        encoded_action_human = np.zeros(13)  # Adjust size based on your task count
        encoded_action_robot = np.zeros(13)

        for task in action[0]:  # Human tasks
            encoded_action_human[encoding_human[task]] = 1
        for task in action[1]:  # Robot tasks
            encoded_action_robot[encoding_robot[task]] = 1

        # Concatenate encoded actions for both human and robot
        return np.concatenate((encoded_action_human, encoded_action_robot))

    def update_value_function(self, state, action, reward):
        """
        Update the value function based on the given state, action, and reward.
        """
        state_tensor = torch.FloatTensor(np.concatenate(([state[0], state[1], state[2]], state[3]))).unsqueeze(0)
        action_tensor = torch.FloatTensor(self.encode_action(action)).unsqueeze(0)
        value = self.value_network(state_tensor, action_tensor)  # Get the current value
        loss = (value - reward) ** 2  # Mean Squared Error
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


if __name__ == '__main__':
    # Define env
    env = gym.make('CollaborationEnv-v0')

    epsilon = 1.0  # Start with full exploration
    epsilon_min = 0.1  # Minimum exploration probability
    epsilon_decay = 0.995  # Decay factor

    EPISODES = 2000
    STEPS = 100

    plt.figure(figsize=(12, 6))
    for SAMPLE_SIZE in [1, 20, 100]:
        # Define agent
        agent = ValueAgent(env, SAMPLE_SIZE)
        times = []
        stress = []
        # Use the environment
        for i_episode in tqdm(range(EPISODES)):
            observation, _ = env.reset()
            stress_peak = 0
            for t in range(STEPS):
                remainingTasks = observation[3]
                action = agent.select_best_action(observation, epsilon)
                # Random agent
                # action = env.action_space.sample(remainingTasks)
                observation, reward, terminated, truncated, info = env.step(action)
                stress_peak = min(stress_peak, reward)

                # Update the value function at each timestep
                agent.update_value_function(observation, action, reward)
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

                # print(f"Action: {action}\nNew State: {observation}\nReward(Stress): {reward}\n")
                if terminated:
                    # print(f"Episode finished after {t + 1} timesteps")
                    times.append(observation[0].item())
                    break

            stress.append(stress_peak)
        # Close env
        env.close()

        # mean_exec_time = np.mean(times)
        # std_exec_time = np.std(times)
        # print(f"Mean exec time: {mean_exec_time}")
        # print(f"Standard Deviation exec time: {std_exec_time}")

        # Calculate running mean of stress
        stress = np.array(stress)
        running_mean_stress = np.cumsum(stress) / (np.arange(1, len(stress) + 1))
        # Smooth the running mean using a moving average
        window_size = 50  # You can adjust the window size for more or less smoothing
        smoothed_running_mean = moving_average(running_mean_stress, window_size)

        # Plot smoothed running mean of stress
        plt.plot(range(len(smoothed_running_mean)), smoothed_running_mean, label=f'action sample size:{SAMPLE_SIZE}')

    dir_path = "../output/value_agent/"
    os.makedirs(dir_path, exist_ok=True)

    plt.xlabel('Episode')
    plt.ylabel('Stress')
    plt.title('Smoothed Running Mean of Stress over Episodes')
    # plt.axhline(y=np.mean(stress), color='red', linestyle='--', label='Baseline Mean')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{dir_path}/smoothed_running_mean_stress.png')
    plt.close()