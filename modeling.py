import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from utils import prep_data
from scipy.stats import gaussian_kde
from scipy.stats import norm


def task_prob_dist(rescheduling = True):

    if rescheduling:
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
        operators = [1, 2, 3, 4, 6, 7]
        dir_path = f"./output/RESCH/probability_estimation/"
    else:
        task_data = {'1': [],
                     '2': [],
                     '3': [],
                     '4': [],
                     '5': [],
                     '6': [],
                     '7': [],
                     '8': [],
                     '9': [],
                     '10': []}
        operators = [1, 2, 3]
        dir_path = f"./output/NO_RESCH/probability_estimation/"
    # Create output destination
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for operator in operators:
        # Get data
        if rescheduling:
            path = './data/csv/RESCH/P0' + str(operator) + '.csv'
        else:
            path = './data/csv/NO_RESCH/P0' + str(operator) + '.csv'
        df = prep_data(path)

        for index, row in df.iterrows():
            tasks = row['opTaskToSave']
            times = row['timetoSave']
            # stress = row['mwtoSave']
            for i in range(len(tasks)):
                task_data[str(tasks[i])].append(times[i])

    with open(f"{dir_path}/gaussian_params.txt", "w") as file:
        bin_count = 30
        for key, times in task_data.items():
            if not times:
                continue
            times = np.array(times)

            plt.hist(times, bins = bin_count, color = 'skyblue', edgecolor = 'black')
            plt.xlabel('Exec time')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Task {key}')

            x_values = np.linspace(min(times), max(times), 1000)

            ### Kernel density estimation (non parametric)
            kde = gaussian_kde(times)
            # Evaluate the KDE at a range of points
            kde_values = kde.evaluate(x_values)
            # Plot the estimated probability density function
            plt.plot(x_values, kde_values, label='Gaussian KDE')

            ### Gaussian MLE (parametric)
            mu, std = norm.fit(times)

            # Save params
            line = f'{key} {mu} {std}\n'
            file.write(line)

            plt.plot(x_values, norm.pdf(x_values, mu, std), label='Gaussian MLE')
            plt.legend()
            # plt.show()
            plt.savefig(f'{dir_path}/task_{key}.png')
            plt.close()

    return task_data


if __name__ == '__main__':
    # task_prob_dist()
    task_prob_dist(rescheduling = False)
