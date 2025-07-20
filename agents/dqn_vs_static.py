import gymnasium as gym
import register_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns


# Top static policies
# 1. Label: (RHRRRH), Avg Reward: -6.15
# 2. Label: (HRHRRH), Avg Reward: -6.16
# 3. Label: (HHHRRR), Avg Reward: -6.18
# 4. Label: (RRHRHH), Avg Reward: -6.18
# 5. Label: (RHHRHR), Avg Reward: -6.19
human_tasks = [1, 2, 3, 4, 5, 6, 10]
robot_tasks = [11, 15, 16, 17, 18, 19, 20]
common_tasks = [7, 8, 9, 12, 13, 14]


def compute_policy(s):
    human_schedule = human_tasks.copy()
    robot_schedule = robot_tasks.copy()
    for i in range(len(s)):
        if s[i] == 'R':
            robot_schedule.append(common_tasks[i])
        elif s[i] == 'H':
            human_schedule.append(common_tasks[i])
        else:
            continue
    return human_schedule, robot_schedule


policy_1 = compute_policy("RHRRRH")
policy_2 = compute_policy("HRHRRH")
policy_3 = compute_policy("HHHRRR")
policy_4 = compute_policy("RRHRHH")
policy_5 = compute_policy("RHHRHR")

mapping_human = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 10: 6, 7: 7, 8: 8, 9: 9, 12: 10, 13: 11, 14: 12}
mapping_robot = {7: 0, 8: 1, 9: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7, 16: 8, 17: 9, 18: 10, 19: 11, 20: 12}

EPISODES = 1000
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
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def action_to_index(action):
    task_idx, actor = action
    return (task_idx - 1) * 2 + actor  # maps (task, actor) to [0, ..., 39]


def index_to_action(index):
    task_idx = index//2 + 1
    actor = index % 2
    return task_idx, actor

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


def save_static_vs_dqn_boxplot(static_rewards, dqn_rewards, labels, filename="../output/plots/static_vs_dqn_boxplot.png"):
    data = []

    # Add static policies
    for i, rewards in enumerate(static_rewards):
        for reward in rewards:
            data.append({'Policy': labels[i], 'Reward': reward})

    # Add DQN
    for reward in dqn_rewards:
        data.append({'Policy': 'DQN', 'Reward': reward})

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        x='Policy',
        y='Reward',
        hue='Policy',
        data=df,
        palette="Set2",
        showmeans=True,
        meanline=True,
        meanprops={"color": "black", "linestyle": "--", "linewidth": 2}
    )

    plt.title("Reward Distributions: Static Policies vs DQN (Mean as Line)")
    plt.ylabel("Reward")
    plt.xlabel("Policy")

    # Remove redundant legend
    if ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Boxplot saved as: {filename}")


env = gym.make('CollaborationEnv-v4')
state = env.reset()[0]
state = flatten_state(state)
input_dim = state.shape[0]
action_space_size = 20 * 2  # 20 tasks × 2 actors

policy_net = DQN(input_dim, action_space_size).to(device)
policy_net.load_state_dict(torch.load("../output/saved_models/dqn_policy_model.pth"))

dqn_rewards = []
p1_rewards = []
p2_rewards = []
p3_rewards = []
p4_rewards = []
p5_rewards = []
static_rewards = [p1_rewards, p2_rewards, p3_rewards, p4_rewards, p5_rewards]
static_schedules = [policy_1, policy_2, policy_3, policy_4, policy_5]
for episode in range(EPISODES):
    state = env.reset()[0]

    human_exec = state['human_exec']
    robot_exec = state['robot_exec']

    for k in range(5):
        human_schedule, robot_schedule = static_schedules[k]
        # Execute human
        possible_task = human_tasks + common_tasks
        time1 = 0
        for task in human_schedule:
            idx = mapping_human[task]
            time1 += human_exec[idx]
        # Execute robot
        possible_task = robot_tasks + common_tasks
        time2 = 0
        for task in robot_schedule:
            idx = mapping_robot[task]
            time2 += robot_exec[idx]
        static_rewards[k].append(max(time1, time2))


    # Run dqn
    state = flatten_state(state)
    steps = 0
    while True:
        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)[0]
            valid_mask = torch.tensor(get_valid_action_mask(env), device=device)
            q_values[~valid_mask] = -1e9
            action = q_values.argmax().item()
        next_state, reward, done, _, _ = env.step(index_to_action(action))
        next_state = flatten_state(next_state)
        state = next_state

        steps += 1
        if done:
            dqn_rewards.append(-reward)
            break


env.close()


average_rewards = {
    i+1: np.mean(rewards)
    for i, rewards in enumerate(static_rewards)
}

print(average_rewards)

print(np.mean(dqn_rewards))

labels = ['(RHRRRH)', '(HRHRRH)', '(HHHRRR)', '(RRHRHH)', '(RHHRHR)']
save_static_vs_dqn_boxplot(static_rewards, dqn_rewards, labels)
