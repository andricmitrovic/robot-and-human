import numpy as np
import joblib


def get_exec_time_params():
    fpath = 'output/RESCH/probability_estimation/gaussian_params.txt'
    data_dict = {}
    with open(fpath, "r") as file:
        for line in file:
            parts = line.split()
            key = int(parts[0])
            value1 = float(parts[1])
            value2 = float(parts[2])
            data_dict[key] = (value1, value2)
    return data_dict


class OperatorGaussian:
    def __init__(self):
        # Get params for Gaussian operator
        self.tasks_params = get_exec_time_params()
        # Stress predictor
        self.stressModel = joblib.load('./output/RESCH/avg_stress/ransac_model.pkl')

    def sample_exec_time(self, currTask):
        mean, std = self.tasks_params[currTask]
        return np.random.normal(mean, std, 1)

    def sample_stress(self, currTime):
        return self.stressModel(currTime)
