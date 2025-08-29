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


class ImprovingOperator:
    def __init__(self, seed=None, improvement_step=0.08):
        """
        improvement_step: fraction of the gap closed towards the min mean at each sample.
        seed: optional for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.tasks_params = get_exec_time_params()
        self.stressModel = joblib.load('../output/RESCH/avg_stress/ransac_model.pkl')

        # Human-possible tasks (including task 12)
        self.human_possible = [1, 2, 3, 4, 5, 6, 10, 7, 8, 9, 12, 13, 14]

        # For each task: original mean, current mean, and minimum target mean
        self.task_means = {}
        self.task_sigmas = {}
        self.task_min_means = {}
        self.improvement_step = improvement_step

        for t in self.human_possible:
            if t == 12:
                # Task 12 starts at fixed mean=0.6 with a small variance
                mu_init = np.log(0.6)
                sigma = 0.05   # << small variance to keep sampling slightly stochastic
            else:
                mu_init, sigma = self.tasks_params[t]

            # Random multiplicator in [0.7, 1.0]
            multiplier = random.uniform(0.7, 1.0)
            min_mu = mu_init * multiplier

            self.task_means[t] = mu_init
            self.task_sigmas[t] = sigma
            self.task_min_means[t] = min_mu

    def sample_exec_time(self, currTask):
        """Sample execution time with improving μ toward its per-task floor."""
        mu = self.task_means[currTask]
        sigma = self.task_sigmas[currTask]
        sample = np.random.lognormal(mean=mu, sigma=sigma, size=1)

        # Update mean toward minimum mean after sampling
        min_mu = self.task_min_means[currTask]
        new_mu = mu - self.improvement_step * (mu - min_mu)
        self.task_means[currTask] = new_mu

        return sample

    def sample_stress(self, currTime):
        return self.stressModel.predict([[currTime]])[0]
