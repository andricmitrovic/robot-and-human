import numpy as np
import joblib

# A simple fake made operator that has constant task execution time and stress level equal to the current time
# Idea is that simple agent will learn to minimize total execution time, and bcs of big exec time for the fake operator
# decide to give him fewer tasks than the robot


class FakeOperator:
    def __init__(self):
        pass

    def sample_exec_time(self, currTask):
        return [2]

    def sample_stress(self, currTime):
        return 0
