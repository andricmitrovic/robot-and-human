import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Sequence, Box, MultiDiscrete
import numpy as np
from operator_sim import OperatorGaussian

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

# (current time, currOperatorTask, currTaskRemaining, (remainingTasks ---> 1...20))

###############
### ACTION ###
# (robot schedule, human schedule)
# !!! actions are done from right to left like a stack


class CollaborationEnv(gym.Env):
    def __init__(self):
        super(CollaborationEnv, self).__init__()

        self.action_space = spaces.Tuple((
            spaces.Sequence(spaces.Box(low=7, high=20, shape=(), dtype=np.int32)),
            # First variable-length integer array
            spaces.Sequence(spaces.Box(low=1, high=14, shape=(), dtype=np.int32))
        # Second variable-length integer array
        ))

        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(), dtype=np.float32),
            spaces.Box(low=0, high=14, shape=(), dtype=np.int32),
            spaces.Box(low=0, high=np.finfo(np.float32).max, shape=(), dtype=np.float32),
            spaces.MultiBinary(20)))

        self.state = self.initState()
        self.operator = OperatorGaussian()

    def initState(self):
        currTime = 0
        currOperatorTask = 0
        currTaskRemaining = 0
        remainingTasks = np.ones(20, dtype=np.int8)
        return currTime, currOperatorTask, currTaskRemaining, remainingTasks

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initState()
        return self.state, {}

    def step(self, action):
        # todo check illegal actions for both
        # todo make sure action is a permutation of remaining tasks and that robot and human can each reach them
        # TODO: remove this an implement your own actions check
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        robotSchedule = list(action[0])
        humanSchedule = list(action[1])

        # Step robot
        task = robotSchedule.pop()
        timePassed = robotExecTime[task]

        # Step human
        doneTasks, currOperatorTask, currTaskRemaining, stress = self.stepHuman(timePassed, humanSchedule)
        stress = stress[0]
        # Update unfinished tasks
        doneTasks.append(task)
        doneTasks.append(currOperatorTask) # it will be done in the future dont assign it, but maybe possible to reassign?
        remainingTasks = self.state[3]
        for idx in doneTasks:
            remainingTasks[idx-1] = 0

        # Check if all task are done
        terminated = np.sum(remainingTasks) == 0
        truncated = False

        # Modify new state
        currTime = self.state[0] + timePassed
        new_state = (currTime, currOperatorTask, currTaskRemaining, remainingTasks)
        self.state = new_state
        # print(new_state)
        # print(self.state)

        return self.state, stress, terminated, truncated, {}

    def stepHuman(self, timePassed, schedule):
        currTime, currOperatorTask, currTaskRemaining, _ = self.state
        # currTask, [currTaskExecTime, startTime] = self.state[1]
        # Just started
        if currOperatorTask == 0:
            currOperatorTask = schedule.pop()
            currTaskRemaining = self.operator.sample_exec_time(currOperatorTask)[0]
        # Finish started tasks
        doneTasks = []
        remaining_time = timePassed

        while currTaskRemaining < remaining_time:
            # Finish the task
            remaining_time -= currTaskRemaining
            doneTasks.append(currOperatorTask)
            # Start a new task
            currTask = schedule.pop()
            currTaskRemaining = self.operator.sample_exec_time(currTask)[0]

        # Finish a part of the task with remaining time
        currTaskRemaining -= remaining_time

        # Sample stress at the end of the step time
        stress = self.operator.sample_stress(currTime+timePassed)
        return doneTasks, currOperatorTask, currTaskRemaining, stress

    def render(self, mode='human', close=False):
        # Implement visualization
        print(f"State: {self.state}")

    def close(self):
        pass
