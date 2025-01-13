import numpy as np
import joblib


def get_exec_time_params():
    fpath = '../output/RESCH/probability_estimation/params_ln.txt'
    data_dict = {}
    with open(fpath, "r") as file:
        for line in file:
            parts = line.split()
            key = int(parts[0])
            value1 = float(parts[1])
            value2 = float(parts[2])
            data_dict[key] = (value1, value2)
    return data_dict


class FakeStressOperator:
    def __init__(self):
        # Get params for Gaussian operator
        self.tasks_params = get_exec_time_params()
        # Stress predictor
        self.stressModel = joblib.load('../output/RESCH/avg_stress/ransac_model.pkl')
        self.stress_slope = 0.08

    def sample_exec_time(self, currTask):
        if currTask == 12: #todo input some smart value for task 12
            return [0.6]
        mu_ln, sigma_ln = self.tasks_params[currTask]
        return np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=1)

    def sample_stress(self, currTime):
        return 20+self.stress_slope*currTime
