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

# (current time, (human info -----> currTask, currTaskExecTime, startTime, stress), (not done tasks ---> 1...20))

###############
### ACTION ###
# (robot schedule, human schedule)
# !!! actions are done from right to left like a stack


class CollaborationEnv(gym.Env):
    def __init__(self):
        super(CollaborationEnv, self).__init__()

        self.action_space = spaces.Tuple((spaces.MultiDiscrete(13), spaces.MultiDiscrete(13))) # or Sequence???

        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=np.finfo(np.float32).max, dtype=np.float32),
            spaces.Tuple((
                        spaces.Box(low=0, high=14, dtype=np.int32),
                        spaces.Box(low=0, high=np.finfo(np.float32).max, shape= (3,), dtype=np.float32))),
            spaces.MultiBinary(20)))

        self.state = self.initState()
        self.operator = OperatorGaussian()

    def initState(self):
        currTime = np.array([0], dtype=np.float32)
        humanInfo = (np.array([0], dtype=np.int32),
                     np.zeros(3, dtype=np.float32),)
        doneTasks = np.ones(20, dtype=np.int8)
        return currTime, humanInfo, doneTasks

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initState()
        return self.state, {}

    def step(self, action):
        # todo check illegal actions for both
        # todo make sure action is a permutation of remaining tasks and that robot and human can each reach them
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Step robot
        task = action[0].pop()
        currTime = self.state[0]
        currTime += robotExecTime[task]

        # Step human
        doneTasks, currHumanTask, stress = self.stepHuman(currTime, action[1])

        # Update unfinished tasks
        doneTasks.append(task)
        doneTasks.append(currHumanTask) # maybe possible to reassign?
        remainingTasks = self.state[2]
        for idx in doneTasks:
            remainingTasks[idx] = 0

        # Check if all task are done
        done = remainingTasks.count(1) == 0

        return self.state, stress, done, {}

    def stepHuman(self, currTime, schedule):
        currTask, [currTaskExecTime, startTime, stress] = self.state[1]
        # Just started
        if currTask == 0:
            currTask = schedule.pop()
            currTaskExecTime = self.operator.sample_exec_time(currTask)
        # Finish started tasks
        doneTasks = []
        while startTime + currTaskExecTime < currTime:
            doneTasks.append(currTask)
            startTime += currTaskExecTime
            # Start a new task
            currTask = schedule.pop()
            currTaskExecTime = self.operator.sample_exec_time(currTask)

        return doneTasks, currTask, self.operator.sample_stress(currTime)

    def render(self, mode='human', close=False):
        # Implement visualization
        print(f"State: {self.state}")

    def close(self):
        pass
