import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from utils import prep_data
from scipy.stats import gaussian_kde
from scipy.stats import norm


def task_prob_dist(plot=True):
    task_data = {'1': [],
                 '2': [],
                 '3': [],
                 '4': [],
                 '5': [],
                 '6': [],
                 '7': [],
                 '8': [],
                 '9': [],
                 '10': [],
                 '12': [],
                 '13': [],
                 '14': []}

    for operator in [1, 2, 3, 4, 6, 7]:
        # Get data
        path = './data/csv/RESCH/P0' + str(operator) + '.csv'
        df = prep_data(path)

        for index, row in df.iterrows():
            tasks = row['opTaskToSave']
            times = row['timetoSave']
            # stress = row['mwtoSave']
            for i in range(len(tasks)):
                task_data[str(tasks[i])].append(times[i])

    if plot:
        bin_count = 30
        # Create output destination
        dir_path = f"./output/exec_time_frequency/operator_{operator}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for key, values in task_data.items():
            if not values:
                continue
            plt.hist(values, bins = bin_count, color = 'skyblue', edgecolor = 'black')
            plt.xlabel('Exec time')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Task {key}')

            x_values = np.linspace(min(values), max(values), 1000)

            ### Kernel density estimation (non parametric)
            kde = gaussian_kde(values)
            # Evaluate the KDE at a range of points
            kde_values = kde.evaluate(x_values)
            # Plot the estimated probability density function
            plt.plot(x_values, kde_values, label='Gaussian KDE')

            ### Gaussian MLE (parametric)
            mu, std = norm.fit(values)
            plt.plot(x_values, norm.pdf(x_values, mu, std), label='Gaussian MLE')

            # Create output destination
            dir_path = f"./output/probability_estimation/"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            plt.legend()
            # plt.show()
            plt.savefig(f'{dir_path}/task_{key}.png')
            plt.close()

    return task_data


if __name__ == '__main__':
    task_prob_dist()
