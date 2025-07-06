import gymnasium as gym
import register_env
import numpy as np

np.random.seed(42)

env = gym.make('CollaborationEnv-v3')
_, _ = env.reset()
env.render()
# action = ([5, 4, 3, 2, 1], [18, 19, 20])
action = env.action_space.sample()
print(action)
observation, reward, terminated, truncated, info = env.step(action)
print(f"Action: {action}\nNew State: {observation}\nReward(Stress): {reward}")

env.close()
