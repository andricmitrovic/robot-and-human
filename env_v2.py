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
    def __init__(self, operator='avg'):
        super(CollaborationEnv_V2, self).__init__()

        self.action_space = VariableLengthActionSpace(low=1, high=20, max_len=20)

        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(), dtype=np.float32),
            spaces.Box(low=-1,  high=1, shape=(20,), dtype=np.int8)))

        self.state = self.initState()
        if operator == 'fake':
            self.operator = FakeOperator()
        elif operator == 'stress+':
            self.operator = FakeStressOperator()
        elif operator == 'avg':
            self.operator = AverageOperator()

        self.reward_coef = [1, 0]

        self.humanExecTime = [self.operator.sample_exec_time(i)[0] for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]]

    def initState(self):
        currTime = np.array(0, dtype=np.float32)
        remainingTasks = np.zeros(20, dtype=np.int8)
        return currTime, remainingTasks

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initState()
        return self.state, {}

    def step(self, action):
        humanSchedule = list(action[0])
        robotSchedule = list(action[1])

        # Execute robot tasks
        robotTime = 0
        for task in robotSchedule:
            robotTime += robotExecTime[task]

        # Execute human tasks
        humanTime = 0
        for task in humanSchedule:
            humanTime += self.humanExecTime[task]

        totalTime = max(humanTime, robotTime)
        idleTime = totalTime - min(humanTime, robotTime)

        # todo: array or one value?
        stress = self.operator.sample_stress(currTime)

        # Check if all task are done
        terminated = True
        truncated = False

        # todo: include idle time?
        reward = -self.reward_coef[0] * currTime + self.reward_coef[1] * stress
        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human', close=False):
        # Implement visualization
        print(f"State: {self.state}")

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

    def sample(self, remaining_tasks):
        # Sample actions for human and robot based on the remaining tasks,
        available_tasks = np.where(remaining_tasks == 0)[0] + 1
        if len(available_tasks) == 0:
            raise ValueError("No available tasks to sample from.")
        # Split available tasks into human, robot, and common tasks
        available_human_tasks = [task for task in available_tasks if task in self.human_tasks]
        available_robot_tasks = [task for task in available_tasks if task in self.robot_tasks]
        available_common_tasks = [task for task in available_tasks if task in self.common_tasks]

        # Randomly assign common tasks to either human or robot
        np.random.shuffle(available_common_tasks)
        len_common_for_human = np.random.randint(0, len(available_common_tasks) + 1)

        # Split the common tasks randomly between human and robot
        common_for_human = available_common_tasks[:len_common_for_human]
        common_for_robot = available_common_tasks[len_common_for_human:]

        # All available human-specific tasks and robot-specific tasks must be included
        human_final_tasks = available_human_tasks + common_for_human
        robot_final_tasks = available_robot_tasks + common_for_robot

        # Shuffle the final tasks for both human and robot
        np.random.shuffle(human_final_tasks)
        np.random.shuffle(robot_final_tasks)

        return (human_final_tasks, robot_final_tasks)

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


