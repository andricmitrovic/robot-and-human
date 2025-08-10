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


class CollaborationEnv_V5(gym.Env):
    def __init__(self, operator='avg'):
        super(CollaborationEnv_V5, self).__init__()

        # Define valid task sets
        self.robot_current = None
        self.human_current = None
        self.robot_completed = None
        self.human_completed = None
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
        self.human_possible = self.human_tasks + self.common_tasks
        self.robot_possible = [key for key, value in robot_exec_time.items()]

        self.all_tasks = self.human_tasks + self.robot_tasks + self.common_tasks
        self.num_tasks = len(self.all_tasks)

        self.action_space = spaces.Discrete(20)

        self.observation_space = spaces.Dict({
            "human_completed": spaces.MultiBinary(13),
            "robot_completed": spaces.MultiBinary(13),
            "human_current": spaces.Discrete(13),
            "robot_current": spaces.Discrete(13),
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
        self.human_completed = np.zeros(13, dtype=np.int8)
        self.robot_completed = np.zeros(13, dtype=np.int8)
        self.remaining_tasks = set(self.all_tasks)
        self.task_pool = np.ones(20, dtype=np.int8)
        self.human_time = 0.0
        self.robot_time = 0.0
        self.episode_log = []
        self.human_current = 0
        self.robot_current = 0
        self.human_exec_time = {
            i: self.operator.sample_exec_time(i)[0] for i in self.human_tasks + self.common_tasks
        }
        self.human_exec = [value for key,value in self.human_exec_time.items()]

        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "human_completed": self.human_completed.copy(),
            "robot_completed": self.robot_completed.copy(),
            "human_current": self.human_current,
            "robot_current": self.robot_current,
            "human_exec": np.array(self.human_exec, dtype=np.float32),
            "robot_exec": np.array(self.robot_exec, dtype=np.float32),
        }

    def action_mapping(self, action, actor):
        if actor:
            idx = self.robot_possible.index(action)
        else:
            idx = self.human_possible.index(action)
        return idx

    def step(self, action):
        task_id = action

        # Check who can do the action and if he is free
        if self.robot_current == 0 and task_id in self.robot_possible:
            actor = 1
        elif self.human_current == 0 and task_id in self.human_possible:
            actor = 0
        elif self.human_current and self.robot_current:
            raise ValueError(f"Both human and robot are busy!")
        else:
            raise ValueError(f"Error for action {task_id}. In state {self._get_obs()}")

        if task_id not in self.remaining_tasks:
            raise ValueError(f"Task {task_id} already completed.")

        if actor == 0 and task_id not in self.human_tasks + self.common_tasks:
            raise ValueError(f"Human cannot do task {task_id}")
        if actor == 1 and task_id not in self.robot_tasks + self.common_tasks:
            raise ValueError(f"Robot cannot do task {task_id}")

        if actor == 0:
            duration = self.human_exec_time[task_id]
            self.human_current = task_id
            # print(self.human_current)
        else:
            duration = robot_exec_time[task_id]
            self.robot_current = task_id

        self.remaining_tasks.remove(task_id)
        self.task_pool[task_id - 1] = 0

        # Finish one queued task
        if self.human_current and self.robot_current:
            if self.human_time < self.robot_time:
                self.execute_task_in_queue(0)
            else:
                self.execute_task_in_queue(1)
        # Check if human or robot is out of tasks to do, then just finish all queued tasks

        if not (self.remaining_tasks & set(self.human_possible)):
            # Human out of tasks to do, execute robot tasks
            self.execute_task_in_queue(1)
        if not (self.remaining_tasks & set(self.robot_possible)):
            # Robot out of tasks, execute human tasks
            self.execute_task_in_queue(0)

        terminated = len(self.remaining_tasks) == 0
        reward = -duration
        # max_time = max(self.human_time, self.robot_time)
        # reward = -max_time
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"Current Time: {max(self.human_time, self.robot_time):.2f}")
        print(f"Remaining Tasks: {sorted(self.remaining_tasks)}")

    def execute_task_in_queue(self, actor):
        if actor:
            self.robot_time += robot_exec_time[self.robot_current]
            self.robot_completed[self.action_mapping(self.robot_current, actor)] = 1
            self.robot_current = 0
        else:
            self.human_time += self.human_exec_time[self.human_current]
            self.human_completed[self.action_mapping(self.human_current, actor)] = 1
            self.human_current = 0

    def close(self):
        pass

    def sample_valid_action(self):
        mask = self.get_action_mask()
        valid_indices = [i + 1 for i, val in enumerate(mask) if val]

        if not valid_indices:
            raise RuntimeError(f"No valid actions available. \n For mask {mask}. \n State {self._get_obs()} \n Remaining: {self.remaining_tasks} \n Task pool: {self.task_pool}")

        return random.choice(valid_indices)

    def get_action_mask(self):
        mask = self.task_pool.copy()
        # If an operator is unavailable, mask his actions
        if self.human_current:
            for action in self.human_tasks:
                mask[action-1] = 0
        if self.robot_current:
            for action in self.robot_tasks:
                mask[action-1] = 0
        return mask


