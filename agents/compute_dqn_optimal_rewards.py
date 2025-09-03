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
model_pth = "../output/saved_models/dqn_policy_model_v7_testing.pth"


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

env = gym.make('CollaborationEnv-v7', operator=operator_type)
state = env.reset()[0]
state = flatten_state(state)
input_dim = state.shape[0]
action_space_size = env.action_space.n  # 6 actions

policy_net = DQN(input_dim, action_space_size).to(device)
policy_net.load_state_dict(torch.load(model_pth))

dqn_rewards = []
optimal_rewards = []

if operator_type == 'avg':
    operator = AverageOperator()
if operator_type == 'noisy':
    operator = NoisyOperator()
if operator_type == 'improving':
    operator = ImprovingOperator()

for episode in range(EPISODES):
    _ = env.reset()
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
            optimal_rewards.append(env.unwrapped.optimal_reward)
            dqn_rewards.append(total_reward)  # negate to match static policy cost
            break

env.close()

np.save(f"../output/dqn_rewards_{operator_type}.npy", dqn_rewards)
np.save(f"../output/optimal_rewards_{operator_type}.npy", optimal_rewards)

