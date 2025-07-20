import gymnasium as gym
from gymnasium import spaces
import numpy as np
from operators.operator_fake import FakeOperator
from operators.operator_sim import AverageOperator
from operators.operator_increasing_stress import FakeStressOperator
import random
import logging

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


class CollaborationEnv_V4(gym.Env):
    def __init__(self, operator='avg'):
        super(CollaborationEnv_V4, self).__init__()

        # Define valid task sets
        self.robot_time = None
        self.human_time = None
        self.episode_log = None
        # self.last_stress = None
        self.current_time = 0
        self.task_pool = None
        self.remaining_tasks = None
        self.human_tasks = [1, 2, 3, 4, 5, 6, 10]
        self.robot_tasks = [11, 15, 16, 17, 18, 19, 20]
        self.common_tasks = [7, 8, 9, 12, 13, 14]

        self.all_tasks = self.human_tasks + self.robot_tasks + self.common_tasks
        self.num_tasks = len(self.all_tasks)

        self.action_space = spaces.Tuple((
            spaces.Discrete(20),
            spaces.Discrete(2),
        ))

        self.observation_space = spaces.Dict({
            "task_pool": spaces.MultiBinary(20),
            "human_time": spaces.Box(low=0, high=1e3, shape=(1,), dtype=np.float32),
            "robot_time": spaces.Box(low=0, high=1e3, shape=(1,), dtype=np.float32),
            "human_exec": spaces.Box(low=0, high=1e3, shape=(13,), dtype=np.float32),
            "robot_exec": spaces.Box(low=0, high=1e3, shape=(13,), dtype=np.float32),
        })

        self.operator = AverageOperator()

        self.reward_coef = [-1.0, 0]  # total time, stress
        self.human_exec_time = {
            i: self.operator.sample_exec_time(i)[0] for i in self.human_tasks + self.common_tasks
        }
        self.human_exec = [value for key,value in self.human_exec_time.items()]
        self.robot_exec = [value for key,value in robot_exec_time.items()]

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.remaining_tasks = set(self.all_tasks)
        self.task_pool = np.ones(20, dtype=np.int8)
        self.human_time = 0.0
        self.robot_time = 0.0
        self.episode_log = []
        self.human_exec_time = {
            i: self.operator.sample_exec_time(i)[0] for i in self.human_tasks + self.common_tasks
        }
        self.human_exec = [value for key,value in self.human_exec_time.items()]

        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "task_pool": self.task_pool.copy(),
            "human_time": np.array([self.human_time], dtype=np.float32),
            "robot_time": np.array([self.robot_time], dtype=np.float32),
            "human_exec": np.array(self.human_exec, dtype=np.float32),
            "robot_exec": np.array(self.robot_exec, dtype=np.float32),
        }

    def step(self, action):
        task_id, actor = action

        if task_id not in self.remaining_tasks:
            raise ValueError(f"Task {task_id} already completed.")

        if actor == 0 and task_id not in self.human_tasks + self.common_tasks:
            raise ValueError(f"Human cannot do task {task_id}")
        if actor == 1 and task_id not in self.robot_tasks + self.common_tasks:
            raise ValueError(f"Robot cannot do task {task_id}")

        if actor == 0:
            duration = self.human_exec_time[task_id]
            self.human_time += duration
            actor_str = "human"
        else:
            duration = robot_exec_time[task_id]
            self.robot_time += duration
            actor_str = "robot"

        self.remaining_tasks.remove(task_id)
        self.task_pool[task_id - 1] = 0
        self.episode_log.append((task_id, actor_str, duration))

        terminated = len(self.remaining_tasks) == 0
        # reward = -duration
        max_time = max(self.human_time, self.robot_time)
        reward = -max_time
        truncated = False
        self.current_time = max(self.human_time, self.robot_time)
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"Current Time: {max(self.human_time, self.robot_time):.2f}")
        print(f"Remaining Tasks: {sorted(self.remaining_tasks)}")

    def close(self):
        pass

    def sample_valid_action(self):
        valid_actions = []

        for task_id in self.remaining_tasks:
            if task_id in self.human_tasks or task_id in self.common_tasks:
                valid_actions.append((task_id, 0))
            if task_id in self.robot_tasks or task_id in self.common_tasks:
                valid_actions.append((task_id, 1))

        if not valid_actions:
            raise RuntimeError("No valid actions available. All tasks may be completed.")

        return random.choice(valid_actions)

    def check_valid_action(self, action):
        task_id, actor = action
        if actor:
            if task_id not in (self.robot_tasks + self.common_tasks):
                return False
            if task_id not in self.remaining_tasks:
                return False
        else:
            if task_id not in (self.human_tasks + self.common_tasks):
                return False
            if task_id not in self.remaining_tasks:
                return False
        return True













# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# from operators.operator_fake import FakeOperator
# from operators.operator_sim import AverageOperator
# from operators.operator_increasing_stress import FakeStressOperator
# import random
# import logging
#
# robot_exec_time = {7: 0.372,
#                  8: 1.1,
#                  9: 0.685,
#                  11: 0.47,
#                  12: 0.489,
#                  13: 0.271,
#                  14: 1.1,
#                  15: 0.62,
#                  16: 0.333,
#                  17: 0.23,
#                  18: 0.878,
#                  19: 0.809,
#                  20: 0.711
#                  }
#
#
# class CollaborationEnv_V3(gym.Env):
#     def __init__(self, operator='avg'):
#         super(CollaborationEnv_V3, self).__init__()
#
#         # Define valid task sets
#         self.task_pool = None
#         self.remaining_tasks = None
#
#         self.human_tasks = [1, 2, 3, 4, 5, 6, 10]
#         self.robot_tasks = [11, 15, 16, 17, 18, 19, 20]
#         self.common_tasks = [7, 8, 9, 12, 13, 14]
#
#         self.all_tasks = self.human_tasks + self.robot_tasks + self.common_tasks
#         self.num_tasks = len(self.all_tasks)
#
#         self.r_done = np.zeros(13, dtype=np.int8)
#         self.o_done = np.zeros(13, dtype=np.int8)
#         self.r_scheduled = 0
#         self.o_scheduled = 0
#
#         self.action_space = spaces.Discrete(20)
#         self.observation_space = spaces.Dict({
#             "r_done": spaces.MultiBinary(13),
#             "o_done": spaces.MultiBinary(13),
#             "r_scheduled": spaces.Discrete(20),
#             "o_scheduled": spaces.Discrete(20),
#             "r_times": spaces.MultiDiscrete([20]*13),
#             "o_times": spaces.MultiBinary([20]*13),
#         })
#
#         self.operator = AverageOperator()
#         self.human_exec_time = {
#             i: self.operator.sample_exec_time(i)[0] for i in self.human_tasks + self.common_tasks
#         }
#         self.reset()
#
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.remaining_tasks = set(self.all_tasks)
#         self.task_pool = np.ones(20, dtype=np.int8)
#
#         self.r_done = np.zeros(13, dtype=np.int8)
#         self.o_done = np.zeros(13, dtype=np.int8)
#         self.r_scheduled = 0
#         self.o_scheduled = 0
#
#         self.human_exec_time = {
#             i: self.operator.sample_exec_time(i)[0] for i in self.human_tasks + self.common_tasks
#         }
#         return self._get_obs(), {}
#
#     def _get_obs(self):
#         return {
#             "r_done": self.r_done,
#             "o_done": self.o_done,
#             "r_scheduled": 0,
#             "o_scheduled": 0,
#             "r_times": self.robot_tasks,
#             "o_times": spaces.MultiBinary([20]*13),
#         }
#
#     def step(self, action):
#         task_id, actor = action
#
#         if task_id not in self.remaining_tasks:
#             raise ValueError(f"Task {task_id} already completed.")
#
#         # Eligibility check
#         if actor == 0 and task_id not in self.human_tasks + self.common_tasks:
#             raise ValueError(f"Human cannot do task {task_id}")
#         if actor == 1 and task_id not in self.robot_tasks + self.common_tasks:
#             raise ValueError(f"Robot cannot do task {task_id}")
#
#         if actor == 0:
#             duration = self.human_exec_time[task_id]
#             self.human_time += duration
#             self.last_stress = self.operator.sample_stress(self.human_time)
#             actor_str = "human"
#         else:
#             duration = robot_exec_time[task_id]
#             self.robot_time += duration
#             actor_str = "robot"
#
#         self.remaining_tasks.remove(task_id)
#         self.task_pool[task_id - 1] = 0         # only for task pool 0-indexed task_id
#         self.episode_log.append((task_id, actor_str, duration))
#
#         terminated = len(self.remaining_tasks) == 0
#         max_time = max(self.human_time, self.robot_time)
#         reward = -duration
#         # reward = -max_time if terminated else 0
#         truncated = False
#         return self._get_obs(), reward, terminated, truncated, {}
#
#     def render(self, mode='human'):
#         print(f"Current Time: {max(self.human_time, self.robot_time):.2f}")#, Last Stress: {self.last_stress:.2f}")
#         print(f"Remaining Tasks: {sorted(self.remaining_tasks)}")
#
#     def close(self):
#         pass
#
#     def sample_valid_action(self):
#         valid_actions = []
#
#         for task_id in self.remaining_tasks:
#             # Check if human can perform the task
#             if task_id in self.human_tasks or task_id in self.common_tasks:
#                 valid_actions.append((task_id, 0))  # 0 = human
#
#             # Check if robot can perform the task
#             if task_id in self.robot_tasks or task_id in self.common_tasks:
#                 valid_actions.append((task_id, 1))  # 1 = robot
#
#         if not valid_actions:
#             raise RuntimeError("No valid actions available. All tasks may be completed.")
#
#         return random.choice(valid_actions)
#
#     def check_valid_action(self, action):
#         task_id, actor = action
#         if actor:
#             if task_id not in (self.robot_tasks + self.common_tasks):
#                 #logging.error(f'Task {task_id} cant be done by the robot')
#                 return False
#             if task_id not in self.remaining_tasks:
#                 #logging.error('Task already done')
#                 return False
#         else:
#             if task_id not in (self.human_tasks + self.common_tasks):
#                 #logging.error(f'Task {task_id} cant be done by the human')
#                 return False
#             if task_id not in self.remaining_tasks:
#                 #logging.error('Task already done')
#                 return False
#         return True
#
#
