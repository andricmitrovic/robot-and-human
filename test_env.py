import gymnasium as gym
import register_env  # Make sure to import your registration module
import numpy as np

np.random.seed(42)

env = gym.make('CollaborationEnv-v0')
observation, _ = env.reset()
env.render()
#
action = ([5, 4, 3, 2, 1], [18, 19, 20])
# # action = env.action_space.sample()
# # print(action)
observation, reward, terminated, truncated, info = env.step(action)
print(f"Action: {action}\nNew State: {observation}\nReward(Stress): {-reward}")

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