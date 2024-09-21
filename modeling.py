import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from utils import prep_data
from scipy.stats import gaussian_kde, lognorm, gamma
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

    file_ln = open(f"{dir_path}/params_ln.txt", "w")
    file_gamma = open(f"{dir_path}/params_gamma.txt", "w")

    bin_count = 30
    for key, times in task_data.items():
        if not times:
            continue
        times = np.array(times)

        plt.hist(times, bins = bin_count, color = 'skyblue', edgecolor = 'black', density=True)
        plt.xlabel('Exec time')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Task {key}')

        x_values = np.linspace(min(times), max(times), 1000)

        ### Log-Normal MLE (parametric)
        shape_ln, loc_ln, scale_ln = lognorm.fit(times, floc=0)  # fit log-normal with fixed location (0)
        mu_ln = np.log(scale_ln)
        sigma_ln = shape_ln

        # Save log-normal params (mu_ln, sigma_ln)
        #line_ln = f'{key} LogNormal mu={mu_ln:.4f} sigma={sigma_ln:.4f}\n'
        line_ln = f'{key} {mu_ln} {sigma_ln}\n'
        file_ln.write(line_ln)

        # Plot the log-normal PDF
        plt.plot(x_values, lognorm.pdf(x_values, shape_ln, loc_ln, scale_ln), label='Log-Normal MLE')

        ### Gamma MLE (parametric)
        shape_g, loc_g, scale_g = gamma.fit(times, floc=0)  # fit gamma with fixed location (0)

        # Save gamma params (shape_g, scale_g)
        # line_g = f'{key} Gamma shape={shape_g:.4f} scale={scale_g:.4f}\n'
        line_g = f'{key} {shape_g} {scale_g}\n'
        file_gamma.write(line_g)

        # Plot the gamma PDF
        plt.plot(x_values, gamma.pdf(x_values, shape_g, loc_g, scale_g), label='Gamma MLE')

        plt.legend()
        plt.savefig(f'{dir_path}/task_{key}.png')
        plt.close()

    file_ln.close()
    file_gamma.close()

    return task_data


if __name__ == '__main__':
    # task_prob_dist()
    task_prob_dist(rescheduling = True)
