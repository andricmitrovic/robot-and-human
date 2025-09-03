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
LR = 1e-3
BATCH_SIZE = 128
BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 5000
EPS_START = 1.0
EPS_END = 0.0
EPS_DECAY_EPISODES = 4000
EPISODES = 10000
SOFT_UPDATE_WEIGHT = 1e-3

# Hyperparam search 1
# LR = 0.0006752145157882616
# BATCH_SIZE = 64
# EPS_END = 0.0008640759735554643
# EPS_DECAY_EPISODES = 2470
# SOFT_UPDATE_WEIGHT = 0.002227329676128557
# EPISODES = 20000
# Best hyperparameters: {'lr': 0.0006752145157882616, 'batch_size': 64, 'eps_start': 0.934795155826079, 'eps_end': 0.0008640759735554643, 'eps_decay_episodes': 2470, 'soft_update_weight': 0.002227329676128557}
# Best reward: 6.116301822065696


# Incremental reward design
# 6.093940310019326 and parameters: {'lr': 0.0009330419170559791, 'batch_size': 256, 'eps_end': 0.065890842084809, 'eps_decay_episodes': 2515, 'episodes': 10000, 'soft_update_weight': 0.011645384572984259, 'hidden_size': 256, 'num_hidden_layers': 2, 'activation': 'LeakyReLU'}. Best is trial 13 with value: 6.093940310019326.
LR = 0.0009330419170559791
BATCH_SIZE = 256
EPS_END = 0.065890842084809
EPS_DECAY_EPISODES = 2515
SOFT_UPDATE_WEIGHT = 0.011645384572984259
EPISODES = 10000
# 6.108287096002524

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_state(state):
    return np.concatenate([
        state["system_time"],
        state["free_agent"],
        state["task_mask"]
    ])


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            # nn.Linear(input_dim, 32),
            # nn.ReLU(),
            # nn.Linear(32, output_dim)

            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
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
        states, actions, rewards, dones, next_states, next_masks = zip(*batch)
        return (
            states,
            actions,
            rewards,
            dones,
            next_states,
            next_masks
        )

    def __len__(self):
        return len(self.buffer)


def buffer_to_tensor(buffer):
    states, actions, rewards, dones, next_states, next_masks = buffer
    state_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    action_tensor = torch.tensor(np.array(actions), dtype=torch.int64).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device)
    dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    next_masks = torch.tensor(np.array(next_masks), dtype=torch.bool).to(device)
    return state_tensor, action_tensor, rewards, dones, next_states, next_masks


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def train():
    env = gym.make('CollaborationEnv-v7')  # Ensure it's registered properly
    state = flatten_state(env.reset()[0])
    input_dim = state.shape[0]
    action_space_size = env.action_space.n  # 6

    policy_net = DQN(input_dim, action_space_size).to(device)
    target_net = DQN(input_dim, action_space_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    buffer = ReplayBuffer(BUFFER_SIZE)
    epsilon = EPS_START

    # Fill replay buffer
    for _ in range(MIN_REPLAY_SIZE):
        action = env.unwrapped.sample_valid_action()
        valid_mask = env.task_mask.copy()
        next_state, reward, done, _, _ = env.step(action)
        next_state = flatten_state(next_state)
        buffer.push((state, action, reward, done, next_state, valid_mask))
        state = next_state if not done else flatten_state(env.reset()[0])

    all_rewards = []
    all_optimal_rewards = []

    for episode in range(EPISODES):
        state = flatten_state(env.reset()[0])
        total_reward = 0
        steps = 0
        optimal_reward = None

        while True:
            # ε-greedy action selection
            if random.random() < epsilon:
                action = env.unwrapped.sample_valid_action()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)[0]

                    # Mask invalid actions
                    valid_mask = torch.tensor(env.task_mask, dtype=torch.bool, device=device)
                    q_values[~valid_mask] = -1e9  # Suppress invalid actions
                    action = q_values.argmax().item()
            valid_mask = env.task_mask.copy()
            next_state, reward, done, _, _ = env.step(action)
            next_state_flat = flatten_state(next_state)
            buffer.push((state, action, reward, done, next_state_flat, valid_mask))
            state = next_state_flat

            steps += 1
            if done:
                # total_reward = -reward
                total_reward = max(env.unwrapped.robot_end_time, env.unwrapped.human_end_time)
                optimal_reward = env.unwrapped.optimal_reward
                break

        # Sample and train from buffer
        states, actions, rewards, dones, next_states, next_masks = buffer_to_tensor(buffer.sample(BATCH_SIZE))

        q_values = policy_net(states).gather(1, actions.unsqueeze(1))
        # with torch.no_grad():
        #     max_next_q_values = target_net(next_states).max(1, keepdim=True)[0]
        #     target_q = rewards + max_next_q_values * (1 - dones)

        with torch.no_grad():
            #  1) action selection with ONLINE net
            q_next_online = policy_net(next_states)
            q_next_online[~next_masks] = -1e9
            next_actions = q_next_online.argmax(dim=1, keepdim=True)

            # 2) action evaluation with TARGET net
            q_next_target = target_net(next_states).gather(1, next_actions)
            q_next_target = q_next_target * (1.0 - dones)
            # Double DQN target
            target_q = rewards + q_next_target

        loss = nn.functional.mse_loss(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        soft_update(target_net, policy_net, SOFT_UPDATE_WEIGHT)

        # Epsilon decay
        if episode < EPS_DECAY_EPISODES:
            epsilon = EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_EPISODES)
        else:
            epsilon = EPS_END

        print(f"Episode {episode+1} | Total reward: {total_reward:.2f} | Steps: {steps} | Epsilon: {epsilon:.3f}")
        all_rewards.append(total_reward)
        all_optimal_rewards.append(optimal_reward)

    env.close()

    os.makedirs("../output/saved_models", exist_ok=True)
    torch.save(policy_net.state_dict(), "../output/saved_models/dqn_policy_model_v7_testing.pth")

    # Plotting
    window_size = 100
    rewards_array = np.array(all_rewards)
    moving_avg = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
    std_dev = np.array([
        rewards_array[max(0, i - window_size):i].std()
        for i in range(window_size, len(rewards_array) + 1)
    ])

    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, label='Moving Average Reward')
    plt.fill_between(range(len(moving_avg)), moving_avg - std_dev, moving_avg + std_dev, alpha=0.3)
    plt.xlabel("Training episode")
    plt.ylabel("Reward")
    plt.title("DQN Reward Progress")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../output/plots/dqn_reward_plot_testing.png")
    plt.close()

    #####
    gap_pct = (np.array(rewards_array) - np.array(all_optimal_rewards)) / np.array(all_optimal_rewards) * 100

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(gap_pct)), gap_pct, s=8, alpha=0.4, label="Episode % gap (DQN vs. optimal)")

    # Optional: add a rolling mean line to show trend
    window = 200
    if len(gap_pct) > window:
        roll = np.convolve(gap_pct, np.ones(window)/window, mode="valid")
        plt.plot(range(window-1, window-1+len(roll)), roll, color="red", linewidth=2, label=f"Rolling mean ({window})")

    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Training episode")
    plt.ylabel("% over optimal (lower is better)")
    plt.title("DQN vs Optimal Policy — Gap Over Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../output/plots/dqn_optimal_gap_scatter.png", dpi=150)
    plt.close()


    print('Mean reward last 1000 episodes:')
    print(np.mean(all_rewards[-1000:]))


if __name__ == "__main__":
    train()
