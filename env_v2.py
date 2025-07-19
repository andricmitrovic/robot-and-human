import gymnasium as gym
from gymnasium import spaces
import numpy as np
from operators.operator_fake import FakeOperator
from operators.operator_sim import AverageOperator
from operators.operator_increasing_stress import FakeStressOperator

robotExecTime = {7: 0.372,
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


class CollaborationEnv_V2(gym.Env):
    def __init__(self, operator='avg', reward_coef=None):
        super(CollaborationEnv_V2, self).__init__()

        if reward_coef is None:
            reward_coef = [1, 0]
        self.action_space = VariableLengthActionSpace(low=1, high=20, max_len=20)

        self.observation_space = spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(2,), dtype=np.float32)

        self.state = None
        if operator == 'fake':
            self.operator = FakeOperator()
        elif operator == 'stress+':
            self.operator = FakeStressOperator()
        elif operator == 'avg':
            self.operator = AverageOperator()

        self.reward_coef = reward_coef

        self.humanExecTime = {i: self.operator.sample_exec_time(i)[0] for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = (0.0, 0.0)
        # resample exec times on reset
        self.humanExecTime = {i: self.operator.sample_exec_time(i)[0] for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]}
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        humanSchedule, robotSchedule = list(action[0]), list(action[1])

        humanTime = sum(self.humanExecTime[task] for task in humanSchedule)
        robotTime = sum(robotExecTime[task] for task in robotSchedule)

        totalTime = max(humanTime, robotTime)
        # idleTime = abs(humanTime - robotTime)

        ### todo change stress sampler to more refined stress
        stress = 0 #self.operator.sample_stress(totalTime)

        # Compute reward as a linear combination
        reward = (
            -self.reward_coef[0] * totalTime -
            self.reward_coef[1] * stress
        )

        terminated = True
        truncated = False

        self.state = (totalTime, stress)

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def render(self, mode='human', close=False):
        # Implement visualization
        print(f"Total Time: {self.state[0]:.2f}, Idle Time: {self.state[1]:.2f}, Stress: {self.state[2]:.2f}")

    def close(self):
        pass


class VariableLengthActionSpace(gym.Space):
    def __init__(self, low, high, max_len):
        # Initialize the bounds for each sequence
        self.low = low
        self.high = high

        # Max lengths for the two sequences
        self.max_len = max_len

        # Legal tasks in each schedule
        self.human_tasks = np.array([i for i in range(1, 11) if i not in [7, 8, 9]])
        self.robot_tasks = np.array([i for i in range(11, 21) if i not in [12, 13, 14]])
        self.common_tasks = np.array([i for i in range(7, 15) if i not in [10, 11]])
        # Define the action space type
        super().__init__(shape=(2,), dtype=np.int32)  # 2 sequences of variable length

    def sample(self):
        # Full task set: sample once per episode
        all_tasks = np.arange(1, 21)
        np.random.shuffle(all_tasks)

        human_final, robot_final = [], []
        for task in all_tasks:
            if task in self.human_tasks:
                human_final.append(task)
            elif task in self.robot_tasks:
                robot_final.append(task)
            elif task in self.common_tasks:
                if np.random.rand() < 0.5:
                    human_final.append(task)
                else:
                    robot_final.append(task)
        return human_final, robot_final

    def contains(self, x):
        # Check if the action x is valid based on the defined task constraints.
        # Check if the given action is a tuple of two sequences
        if not isinstance(x, tuple) or len(x) != 2:
            return False

        seq_1, seq_2 = x  # seq_1: human tasks, seq_2: robot tasks

        # Ensure both sequences are lists
        if not isinstance(seq_1, list) or not isinstance(seq_2, list):
            return False

        # Check if all human tasks are either in human_tasks or common_tasks
        if not all(task in self.human_tasks or task in self.common_tasks for task in seq_1):
            return False

        # Check if all robot tasks are either in robot_tasks or common_tasks
        if not all(task in self.robot_tasks or task in self.common_tasks for task in seq_2):
            return False

        # Ensure there is no overlap between human and robot tasks
        if set(seq_1) & set(seq_2):  # Check for any common tasks appearing in both lists
            return False

        return True

    def __repr__(self):
        return f"VariableLengthActionSpace(({self.low}, {self.high}))"


