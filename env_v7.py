import gymnasium as gym
from gymnasium import spaces
import numpy as np
from operators.operator_fake import FakeOperator
from operators.operator_sim import AverageOperator
from operators.operator_noisy import NoisyOperator
from operators.operator_improving import ImprovingOperator
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


class CollaborationEnv_V7(gym.Env):
    def __init__(self, operator='avg'):
        super(CollaborationEnv_V7, self).__init__()

        # Define valid task sets
        self.robot_end_time = None
        self.human_end_time = None

        # self.last_stress = None
        self.current_time = 0
        self.task_mask = None
        self.system_time = None
        self.free_agent = None

        self.human_tasks = [1, 2, 3, 4, 5, 6, 10]
        self.robot_tasks = [11, 15, 16, 17, 18, 19, 20]
        self.common_tasks = [7, 8, 9, 12, 13, 14]
        self.human_possible = self.human_tasks + self.common_tasks
        self.robot_possible = [key for key, value in robot_exec_time.items()]

        self.all_tasks = self.human_tasks + self.robot_tasks + self.common_tasks
        self.num_tasks = len(self.all_tasks)

        self.action_space = spaces.Discrete(6)

        self.observation_space = spaces.Dict({
            "system_time": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "free_agent": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            "task_mask": spaces.MultiBinary(6),
            # "human_exec": spaces.Box(low=0, high=1e3, shape=(13,), dtype=np.float32),
            # "robot_exec": spaces.Box(low=0, high=1e3, shape=(13,), dtype=np.float32),
        })

        self.operator_type = operator
        self.operator = None
        self.task_noise = None
        self.step_counter = 0

        if operator == 'avg':
            self.operator = AverageOperator()
        elif operator == 'noisy':
            self.operator = NoisyOperator()
        elif operator == 'improving':
            self.operator = ImprovingOperator()

        if self.operator is None:
            raise ValueError('Operator model not recognized!')

        self.reward_coef = [-1.0, 0]  # total time, stress
        # self.human_exec_time = {
        #     i: self.operator.sample_exec_time(i)[0] for i in self.human_tasks + self.common_tasks
        # }
        # self.human_exec = [value for key,value in self.human_exec_time.items()]
        # self.robot_exec = [value for key,value in robot_exec_time.items()]

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.human_end_time, self.robot_end_time = self.execute_uncommon_tasks()

        self.system_time = min(self.human_end_time, self.robot_end_time)
        self.free_agent = 0 if self.human_end_time < self.robot_end_time else 1
        self.task_mask = np.ones(6, dtype=np.int8)
        self.step_counter += 1
        if self.operator_type == 'noisy':
            self.operator = NoisyOperator()
        if self.operator_type == 'improving' and self.step_counter > 20:
            self.operator = ImprovingOperator()
            self.step_counter = 0
        # todo need to explicitly say when we changed operators
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "system_time": np.array([self.system_time], dtype=np.float32),
            "free_agent": np.array([self.free_agent], dtype=np.int8),
            "task_mask": self.task_mask
            # "human_exec": np.array(self.human_exec, dtype=np.float32),
            # "robot_exec": np.array(self.robot_exec, dtype=np.float32),
        }

    def execute_uncommon_tasks(self):
        human_end_time = 0
        robot_end_time = 0

        for task in self.human_tasks:
            human_end_time += self.operator.sample_exec_time(task)[0]
        for task in self.robot_tasks:
            robot_end_time += robot_exec_time[task]

        return human_end_time, robot_end_time

    def action_mapping(self, action):
        idx = self.common_tasks.index(action)
        return idx

    def inverse_action_mapping(self, idx):
        task = self.common_tasks[idx]
        return task

    def step(self, idx):
        if not self.task_mask[idx]:
            raise ValueError(f"Action already done!")

        self.task_mask[idx] = 0
        task = self.inverse_action_mapping(idx)

        if self.free_agent == 0:
            duration = self.operator.sample_exec_time(task)[0]
            self.human_end_time += duration
        else:
            duration = robot_exec_time[task]
            self.robot_end_time += duration

        if self.human_end_time < self.robot_end_time:
            self.free_agent = 0
        else:
            self.free_agent = 1

        self.system_time = min(self.robot_end_time, self.human_end_time)

        if np.all(self.task_mask == 0):
            terminated = True
        else:
            # self.system_time = max(self.robot_end_time, self.human_end_time)
            terminated = False
        reward = -max(self.robot_end_time, self.human_end_time)
        # max_time = max(self.human_time, self.robot_time)
        # reward = -self.system_time
        return self._get_obs(), reward, terminated, False, {}

    def render(self, mode='human'):
        print(f"Current Time: {self.system_time:.2f}")
        print(f"Remaining Tasks: {self.task_mask}")

    def close(self):
        pass

    def sample_valid_action(self):
        valid_indices = [i for i, val in enumerate(self.task_mask) if val]
        if not valid_indices:
            raise RuntimeError(f"No valid actions available. \n State {self._get_obs()} \n")

        return random.choice(valid_indices)


