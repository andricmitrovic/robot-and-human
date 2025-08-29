import numpy as np
import random
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


class NoisyOperator:
    def __init__(self, prob_noise=1, max_noise=0.5, min_noise=0, seed=None):
        """
        prob_noise: probability that a human-possible task gets noise
        max_noise:  maximum noise fraction (e.g., 0.30 => up to ±30%)
        min_noise:  minimum noise fraction when chosen
        seed:       optional seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.tasks_params = get_exec_time_params()
        self.stressModel = joblib.load('../output/RESCH/avg_stress/ransac_model.pkl')

        # Human-possible tasks (human + common)
        self.human_possible = [1, 2, 3, 4, 5, 6, 10, 7, 8, 9, 12, 13, 14]

        # Fixed noise map per operator
        self.task_noise_map = self._make_task_noise_map(prob_noise, min_noise, max_noise)

    def _make_task_noise_map(self, prob_noise, min_noise, max_noise):
        """Assign each human task a fixed % noise (positive number)."""
        noise_map = {}
        for t in self.human_possible:
            if random.random() < prob_noise:
                noise_map[t] = round(random.uniform(min_noise, max_noise), 3)
            else:
                noise_map[t] = 0.0
        return noise_map

    def sample_exec_time(self, currTask):
        # Baseline sample
        if currTask == 12:
            sample = np.array([0.6])
        else:
            mu_ln, sigma_ln = self.tasks_params[currTask]
            sample = np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=1)

        # Apply ±% noise
        noise_frac = self.task_noise_map.get(currTask, 0.0)
        if noise_frac > 0.0:
            direction = random.choice([-1, 1])  # randomly increase or decrease
            sample = sample * (1.0 + direction * noise_frac)

        return sample

    def sample_stress(self, currTime):
        return self.stressModel.predict([[currTime]])[0]
