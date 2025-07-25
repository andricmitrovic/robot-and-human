import gymnasium as gym
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
from gymnasium import spaces
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

# ✅ Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ✅ Import your custom environment
import register_env

# ✅ Flatten tuple action space → Discrete(40)
class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_action_space = spaces.Tuple((spaces.Discrete(20), spaces.Discrete(2)))
        self.action_space = spaces.Discrete(40)  # Flattened: task × actor
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "action_mask": spaces.MultiBinary(40)
        })

    def action(self, action_idx):
        task_idx = action_idx // 2
        actor = action_idx % 2
        return (task_idx + 1, actor)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["action_mask"] = self.get_valid_action_mask()
        return obs, info

    def step(self, action_idx):
        action = self.action(action_idx)
        obs, reward, done, truncated, info = self.env.step(action)
        obs["action_mask"] = self.get_valid_action_mask()
        return obs, reward, done, truncated, info

    def get_valid_action_mask(self):
        mask = np.zeros(40, dtype=bool)
        for task_idx in range(20):
            task_id = task_idx + 1
            if self.env.task_pool[task_idx] == 0:
                continue
            for actor in [0, 1]:
                if self.env.check_valid_action((task_id, actor)):
                    mask[(task_id - 1) * 2 + actor] = 1
        return mask


# === Custom Masked Categorical Distribution ===
class MaskedCategoricalDistribution(CategoricalDistribution):
    def proba_distribution(self, action_logits, action_mask=None):
        if action_mask is not None:
            action_logits = action_logits.clone()
            invalid_mask = ~action_mask
            action_logits[invalid_mask] = -1e8
        return super().proba_distribution(action_logits)

    def log_prob(self, actions):
        return super().log_prob(actions)

    def entropy(self):
        return super().entropy()

    def sample(self):
        return super().sample()

# === Custom Policy with Action Masking ===
class MaskedPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace standard distribution with masked version
        self.action_dist = MaskedCategoricalDistribution(self.action_dist.param_shape, self.action_dist.dtype, self.device)

    def forward(self, obs, deterministic=False):
        mask = obs["action_mask"].bool()
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution = self.action_dist.proba_distribution(distribution.distribution.logits, action_mask=mask)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def _predict(self, obs, deterministic=False):
        return self.forward(obs, deterministic)[0]


# ✅ Environment factory
def make_env(log_dir=None):
    env = gym.make('CollaborationEnv-v3')
    env = PPOCollabWrapper(env)
    if log_dir:
        env = Monitor(env, log_dir)
    return env

# ✅ Setup training
def train():
    log_dir = "./ppo_logs/"
    env = DummyVecEnv([lambda: make_env(log_dir)])
    # check_env(make_env())  # Optional: validate env conforms to Gym API

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=128,
        n_steps=2048,
        gamma=1.0
    )

    # ✅ Train
    model.learn(total_timesteps=500_000)
    model.save("ppo_collab_model")

    return log_dir

# ✅ Plot rewards from Monitor logs
def plot_rewards(log_dir, output_path="output/plots/ppo_reward_plot.png", window_size=50):
    df = pd.read_csv(log_dir + "monitor.csv", skiprows=1)
    rewards = df["r"].to_numpy()

    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    std_dev = np.array([
        rewards[max(0, i - window_size):i].std()
        for i in range(window_size, len(rewards) + 1)
    ])

    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, label="Moving Average Reward")
    plt.fill_between(range(len(moving_avg)), moving_avg - std_dev, moving_avg + std_dev, alpha=0.3)
    plt.xlabel("Training Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training - Average Total Reward per Episode")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# ✅ Run everything
if __name__ == "__main__":
    log_dir = train()
    plot_rewards(log_dir)
