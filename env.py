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

###############
### STATE ###
# v_0
# (current time, currOperatorTask, currTaskRemaining, remainingTasks ---> 1...20)

# v_1
# (current time, remainingTasks ---> 1...20)

###############
### ACTION ###
# (humanSchedule, robotSchedule)
# actions are done from right to left


class CollaborationEnv(gym.Env):
    def __init__(self, operator='avg'):
        super(CollaborationEnv, self).__init__()

        self.action_space = VariableLengthActionSpace(low=1, high=20, max_len=20)

        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(), dtype=np.float32),
            spaces.Box(low=0, high=14, shape=(), dtype=np.int32),
            spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(), dtype=np.float32),
            spaces.MultiBinary(20)))

        self.state = self.initState()
        if operator == 'fake':
            self.operator = FakeOperator()
        elif operator == 'stress+':
            self.operator = FakeStressOperator()
        elif operator == 'avg':
            self.operator = AverageOperator()

        self.reward_coef = [1, 0] #[0.8, 0.2]

    def initState(self):
        currTime = np.array(0, dtype=np.float32)
        currOperatorTask = np.array(0, dtype=np.int32)
        currTaskRemaining = np.array(0, dtype=np.float32)
        remainingTasks = np.zeros(20, dtype=np.int8)
        return currTime, currOperatorTask, currTaskRemaining, remainingTasks

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initState()
        return self.state, {}

    def step(self, action):
        humanSchedule = list(action[0])
        robotSchedule = list(action[1])

        # Step robot
        if len(robotSchedule) > 0:
            task = robotSchedule.pop()
            timePassed = robotExecTime[task]
        else:
            task = None
            timePassed = 0.4

        # Step human
        doneTasks, currOperatorTask, currTaskRemaining, stress = self.stepHuman(timePassed, humanSchedule)
        # Update unfinished tasks
        if task is not None:
            doneTasks.append(task)
        doneTasks.append(currOperatorTask)
        remainingTasks = self.state[3]
        for idx in doneTasks:
            remainingTasks[idx-1] = 1

        # Check if all task are done
        terminated = np.count_nonzero(remainingTasks) == 20
        truncated = False

        # Modify new state
        currTime = self.state[0] + timePassed
        self.state = (
            np.array(currTime, dtype=np.float32),
            np.array(currOperatorTask, dtype=np.int32),
            np.array(currTaskRemaining, dtype=np.float32),
            np.array(remainingTasks, dtype=np.int8)
        )
        reward = -self.reward_coef[0] * currTime + self.reward_coef[1] * stress
        return self.state, reward, terminated, truncated, {}

    def stepHuman(self, timePassed, schedule):
        currTime, currOperatorTask, currTaskRemaining, _ = self.state
        # Check if no scheduled tasks
        if len(schedule) == 0:
            return [], 0, 0, -self.operator.sample_stress(currTime+timePassed)
        # Start the first task for human if none is assigned
        if currOperatorTask == 0:
            currOperatorTask = schedule.pop()
            currTaskRemaining = self.operator.sample_exec_time(currOperatorTask)[0]

        doneTasks = []
        remaining_time = timePassed
        # Process tasks for human operator
        while currTaskRemaining < remaining_time:
            remaining_time -= currTaskRemaining
            doneTasks.append(currOperatorTask)

            if len(schedule) == 0:
                currOperatorTask = 0
                currTaskRemaining = 0
                break
            else:
                currOperatorTask = schedule.pop()
                currTaskRemaining = self.operator.sample_exec_time(currOperatorTask)[0]

        # Finish a part of the task with remaining time
        currTaskRemaining -= remaining_time

        # Sample stress at the end of the step time
        stress = -self.operator.sample_stress(currTime+timePassed)
        return doneTasks, currOperatorTask, currTaskRemaining, stress

    def render(self, mode='human', close=False):
        # Implement visualization
        print(f"State: {self.state}")

    def close(self):
        pass


class CollaborationEnv_V1(gym.Env):
    def __init__(self, operator='avg'):
        super(CollaborationEnv_V1, self).__init__()

        self.action_space = VariableLengthActionSpace(low=1, high=20, max_len=20)

        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(), dtype=np.float32),
            spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)))

        self.state = self.initState()
        if operator == 'fake':
            self.operator = FakeOperator()
        elif operator == 'stress+':
            self.operator = FakeStressOperator()
        elif operator == 'avg':
            self.operator = AverageOperator()

        self.reward_coef = [1, 0] #[0.8, 0.2]

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
        currTime, remainingTasks = self.state
        humanSchedule = list(action[0])
        robotSchedule = list(action[1])

        # Step robot
        if len(robotSchedule) > 0:
            task = robotSchedule.pop()
            timePassed = robotExecTime[task]
            remainingTasks[task-1] = 1
        else:
            task = None
            timePassed = 0.4

        currTime = self.state[0] + timePassed

        # Step human
        remainingTasks = self.stepHuman(currTime, remainingTasks, humanSchedule)
        stress = self.operator.sample_stress(currTime)

        # Check if all task are done
        terminated = np.count_nonzero(remainingTasks) == 20
        truncated = False

        # Modify new state
        self.state = (
            np.array(currTime, dtype=np.float32),
            np.array(remainingTasks, dtype=np.int8)
        )
        reward = -self.reward_coef[0] * currTime + self.reward_coef[1] * stress
        return self.state, reward, terminated, truncated, {}

    def stepHuman(self, currTime, remainingTasks, humanSchedule):
        completedHuman = [i for i, task in enumerate(remainingTasks) if task == -1]
        completedTime = sum(self.humanExecTime[i] for i in completedHuman)
        deltaTime = currTime - completedTime

        # Check if no scheduled tasks
        if len(humanSchedule) == 0:
            return remainingTasks

        # Start the first task for human if none is assigned
        currTask = humanSchedule[-1]
        taskTime = self.humanExecTime[currTask-1]

        while deltaTime - taskTime >= 0:
            humanSchedule.pop()
            deltaTime -= taskTime
            remainingTasks[currTask-1] = -1

            if len(humanSchedule) == 0:
                break
            else:
                currTask = humanSchedule[-1]
                taskTime = self.humanExecTime[currTask-1]

        return remainingTasks

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


if __name__ == '__main__':
    # Example usage:
    action_space = VariableLengthActionSpace(low=1, high=20, max_len=20)

    # Sampling a random action
    remaining_tasks = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    random_action = action_space.sample(remaining_tasks)
    print("Random action:", random_action)

    # Check if a specific action is valid
    valid = action_space.contains(([5, 4, 3, 2, 1], [18, 19, 20]))
    print("Is action valid?", valid)
