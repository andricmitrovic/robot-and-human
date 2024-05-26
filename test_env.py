import gymnasium as gym
import register_env  # Make sure to import your registration module
import numpy as np


env = gym.make('CollaborationEnv-v0')
observation, _ = env.reset()
print(observation)

# # Use the environment
# for i_episode in range(10):
#     observation = env.reset()
#     for t in range(10):
#         env.render()
#         action = [env.action_space.space.sample() for _ in range(np.random.randint(1, 5))]  # take a random sequence of actions
#         observation, reward, done, info = env.step(action)
#         print(f"Action: {action}, New State: {observation}, Reward: {reward}")
#         if done:
#             print(f"Episode finished after {t+1} timesteps")
#             break
env.close()