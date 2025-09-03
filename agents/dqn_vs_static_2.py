import gymnasium as gym
import register_env
import numpy as np
import torch
import torch.nn as nn
import random
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from operators.operator_sim import AverageOperator
from operators.operator_noisy import NoisyOperator
from operators.operator_improving import ImprovingOperator

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Set operator type for which you are running a comparison
operator_type = 'improving'
model_pth = "../output/saved_models/dqn_policy_model_v7_avg.pth"

# Top static policies
# 1. Label: (RHRRRH), Avg Reward: -6.15
# 2. Label: (HRHRRH), Avg Reward: -6.16
# 3. Label: (HHHRRR), Avg Reward: -6.18
# 4. Label: (RRHRHH), Avg Reward: -6.18
# 5. Label: (RHHRHR), Avg Reward: -6.19
human_tasks = [1, 2, 3, 4, 5, 6, 10]
robot_tasks = [11, 15, 16, 17, 18, 19, 20]
common_tasks = [7, 8, 9, 12, 13, 14]

robot_exec_time = {7: 0.372,
                 8: 1.1,
                 9: 0.685,
                 11: 0.47,
                 12: 0.489,
                 13: 0.271,
                 14: 1.1,
                 15: 0.62,
                 16: 0.333,
                 17: 0.23,
                 18: 0.878,
                 19: 0.809,
                 20: 0.711
                 }


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

EPISODES = 10000
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


def save_static_vs_dqn_boxplot(static_rewards, dqn_rewards, labels, filename=f"../output/plots/static_vs_dqn_boxplot_{operator_type}.png"):
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


env = gym.make('CollaborationEnv-v7', operator=operator_type)
state = env.reset()[0]
state = flatten_state(state)
input_dim = state.shape[0]
action_space_size = env.action_space.n  # 6 actions

policy_net = DQN(input_dim, action_space_size).to(device)
policy_net.load_state_dict(torch.load(model_pth))

dqn_rewards = []
p1_rewards = []
p2_rewards = []
p3_rewards = []
p4_rewards = []
p5_rewards = []
static_rewards = [p1_rewards, p2_rewards, p3_rewards, p4_rewards, p5_rewards]
static_schedules = [policy_1, policy_2, policy_3, policy_4, policy_5]


if operator_type == 'avg':
    operator = AverageOperator()
if operator_type == 'noisy':
    operator = NoisyOperator()
if operator_type == 'improving':
    operator = ImprovingOperator()

for episode in range(EPISODES):
    _ = env.reset()

    # Run static policies
    for k in range(5):
        human_schedule, robot_schedule = static_schedules[k]
        time1 = sum(operator.sample_exec_time(task)[0] for task in human_schedule)
        time2 = sum(robot_exec_time[task] for task in robot_schedule)
        static_rewards[k].append(max(time1, time2))

    # Run DQN
    state = flatten_state(env.reset()[0])
    while True:
        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)[0]

            valid_mask = torch.tensor(env.task_mask, dtype=torch.bool, device=device)
            q_values[~valid_mask] = -1e9  # mask invalid actions

            action = q_values.argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        state = flatten_state(next_state)

        if done:
            # total_reward = -reward
            total_reward = max(env.unwrapped.robot_end_time, env.unwrapped.human_end_time)
            dqn_rewards.append(total_reward)  # negate to match static policy cost
            break

env.close()

# Average rewards
average_rewards = {
    f"Policy {i+1}": np.mean(rewards)
    for i, rewards in enumerate(static_rewards)
}
print("Static policy average rewards:")
print(average_rewards)

print("\nDQN average reward:")
print(np.mean(dqn_rewards))

# Save boxplot
labels = ['(RHRRRH)', '(HRHRRH)', '(HHHRRR)', '(RRHRHH)', '(RHHRHR)']
save_static_vs_dqn_boxplot(static_rewards, dqn_rewards, labels)

np.save(f"../output/static_rewards_{operator_type}.npy", static_rewards[0])
np.save(f"../output/dqn_rewards_{operator_type}.npy", dqn_rewards)
