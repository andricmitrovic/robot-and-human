import gymnasium as gym
import register_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Hyperparameters
# LR = 1e-3
# BATCH_SIZE = 128
BUFFER_SIZE = 100000
# MIN_REPLAY_SIZE = 5000
EPS_START = 1.0
EPS_END = 0.0
# EPS_DECAY_EPISODES = 4000
EPISODES = 10000
# SOFT_UPDATE_WEIGHT = 1e-3

LR = 5e-4
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 10000
EPS_DECAY_EPISODES = 3000
SOFT_UPDATE_WEIGHT = 5e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_state(state):
    return np.concatenate([
        state["task_pool"],
        state["human_time"],
        state["robot_time"],
        state["human_exec"],
        state["robot_exec"]
    ])


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)
        return (
            states,
            actions,
            rewards,
            dones,
            next_states
        )

    def __len__(self):
        return len(self.buffer)


def action_to_index(action):
    task_idx, actor = action
    return (task_idx - 1) * 2 + actor  # maps (task, actor) to [0, ..., 39]


def index_to_action(index):
    task_idx = index//2 + 1
    actor = index % 2
    return task_idx, actor


def buffer_to_tensor(buffer):
    states, actions, rewards, dones, next_states = buffer
    state_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    action_tensor = torch.tensor(np.array(actions), dtype=torch.int64).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device)
    dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    return state_tensor, action_tensor, rewards, dones, next_states


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def get_valid_action_mask(env):
    mask = np.zeros(40, dtype=bool)
    for task_idx in range(20):
        task_id = task_idx + 1
        if env.unwrapped.task_pool[task_idx] == 0:
            continue  # Task already completed

        for actor in [0, 1]:  # 0 = human, 1 = robot
            if env.unwrapped.check_valid_action((task_id, actor)):
                flat_index = action_to_index((task_id, actor))
                mask[flat_index] = True
    return mask


def train():
    env = gym.make('CollaborationEnv-v4')
    state = env.reset()[0]
    state = flatten_state(state)
    input_dim = state.shape[0]
    action_space_size = 20 * 2  # 20 tasks × 2 actors

    policy_net = DQN(input_dim, action_space_size).to(device)
    target_net = DQN(input_dim, action_space_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    buffer = ReplayBuffer(BUFFER_SIZE)
    epsilon = EPS_START

    # Fill replay buffer
    for _ in range(MIN_REPLAY_SIZE):
        action = env.unwrapped.sample_valid_action()
        next_state, reward, done, _, _ = env.step(action)
        next_state = flatten_state(next_state)
        buffer.push((state, action_to_index(action), reward, done, next_state))
        state = next_state if not done else flatten_state(env.reset()[0])

    all_rewards = []
    # Training loop
    for episode in range(EPISODES):
        state = flatten_state(env.reset()[0])
        total_reward = 0
        steps = 0

        while True:
            action = None
            if random.random() < epsilon:
                action = action_to_index(env.unwrapped.sample_valid_action())
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)[0]
                    valid_mask = torch.tensor(get_valid_action_mask(env), device=device)
                    q_values[~valid_mask] = -1e9
                    action = q_values.argmax().item()
            next_state, reward, done, _, _ = env.step(index_to_action(action))
            next_state = flatten_state(next_state)
            buffer.push((state, action, reward, done, next_state))
            state = next_state
            # total_reward += reward
            if done:
                total_reward = reward
            #     reward += total_reward

            steps += 1

            # Train
            if steps % 4 == 0:
                states, actions, rewards, dones, next_states = buffer_to_tensor(buffer.sample(BATCH_SIZE))

                q_values = policy_net(states).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + max_next_q_values * (1 - dones)

                loss = nn.functional.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        soft_update(target_net, policy_net, SOFT_UPDATE_WEIGHT)

        if episode < EPS_DECAY_EPISODES:
            epsilon = EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_EPISODES)
        else:
            epsilon = EPS_END
        print(f"Episode {episode+1} | Total reward: {total_reward:.2f} | Steps: {steps} | Epsilon: {epsilon:.3f}")
        all_rewards.append(total_reward)

    env.close()

    os.makedirs("../output/saved_models", exist_ok=True)
    torch.save(policy_net.state_dict(), "../output/saved_models/dqn_policy_model.pth")

    # Compute moving average and standard deviation
    window_size = 50
    rewards_array = np.array(all_rewards)
    moving_avg = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
    std_dev = np.array([
        rewards_array[max(0, i - window_size):i].std()
        for i in range(window_size, len(rewards_array) + 1)
    ])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, label='Moving Average Reward')
    plt.fill_between(range(len(moving_avg)), moving_avg - std_dev, moving_avg + std_dev, alpha=0.3)
    plt.xlabel("Training episode")
    plt.ylabel("Score (min)")
    plt.title("Average Total Reward per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../output/plots/dqn_reward_plot.png")  # Save the plot if needed
    #plt.show()


if __name__ == "__main__":
    train()
