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
import optuna

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial):
    # --- Suggest training hyperparameters ---
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    eps_end = trial.suggest_uniform('eps_end', 0.0, 0.2)
    eps_decay = trial.suggest_int('eps_decay_episodes', 500, 5000)
    episodes = trial.suggest_categorical('episodes', [4000, 6000, 8000, 10000])
    soft_update_weight = trial.suggest_loguniform('soft_update_weight', 1e-4, 5e-2)

    # --- Suggest model architecture ---
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 3)
    activation_choice = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'Tanh'])

    # Map activation choice to PyTorch layer
    activation_map = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
        'Tanh': nn.Tanh()
    }
    activation_fn = activation_map[activation_choice]

    # --- Inject into globals for training ---
    global LR, BATCH_SIZE, EPS_START, EPS_END, EPS_DECAY_EPISODES, SOFT_UPDATE_WEIGHT, EPISODES
    LR = lr
    BATCH_SIZE = batch_size
    EPS_END = eps_end
    EPISODES = episodes
    EPS_DECAY_EPISODES = eps_decay
    SOFT_UPDATE_WEIGHT = soft_update_weight

    # Redefine DQN with tuned architecture
    class TunedDQN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            layers = [nn.Linear(input_dim, hidden_size), activation_fn]
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(activation_fn)
            layers.append(nn.Linear(hidden_size, output_dim))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    # Patch global DQN class reference
    global DQN
    DQN = TunedDQN

    # --- Run a shorter training to evaluate ---
    rewards = run_training_for_tuning(episodes=EPISODES)  # shorter training for speed

    return np.mean(rewards[-1000:])  # Mean reward over last 1000 episodes


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
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.LeakyReLU(),
            nn.Linear(32, output_dim)
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


def run_training_for_tuning(episodes=500):
    # A shortened version of your train() loop
    import gymnasium as gym
    import torch
    import random
    from collections import deque
    import numpy as np

    env = gym.make('CollaborationEnv-v7')
    state = flatten_state(env.reset()[0])
    input_dim = state.shape[0]
    action_space_size = env.action_space.n

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
        buffer.push((state, action, reward, done, next_state))
        state = next_state if not done else flatten_state(env.reset()[0])

    all_rewards = []

    for episode in range(episodes):
        state = flatten_state(env.reset()[0])
        total_reward = 0

        while True:
            if random.random() < epsilon:
                action = env.unwrapped.sample_valid_action()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)[0]
                    valid_mask = torch.tensor(env.task_mask, dtype=torch.bool, device=device)
                    q_values[~valid_mask] = -1e9
                    action = q_values.argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            next_state_flat = flatten_state(next_state)
            buffer.push((state, action, reward, done, next_state_flat))
            state = next_state_flat
            if done:
                total_reward = -reward
                break

        states, actions, rewards, dones, next_states = buffer_to_tensor(buffer.sample(BATCH_SIZE))
        q_values = policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            max_next_q_values = target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + max_next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        soft_update(target_net, policy_net, SOFT_UPDATE_WEIGHT)

        if episode < EPS_DECAY_EPISODES:
            epsilon = EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_EPISODES)
        else:
            epsilon = EPS_END

        all_rewards.append(total_reward)

    env.close()
    return all_rewards


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best hyperparameters:", study.best_params)
    print("Best reward:", study.best_value)
