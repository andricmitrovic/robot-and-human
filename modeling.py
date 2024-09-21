import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from utils import prep_data
from scipy.stats import lognorm, gamma


def task_prob_dist(rescheduling=True):
    if rescheduling:
        task_data = {str(i): [] for i in range(1, 15) if i not in [11]}  # Tasks 1 to 14 except 11
        operators = [1, 2, 3, 4, 6, 7]
        dir_path = "./output/RESCH/probability_estimation/"
    else:
        task_data = {str(i): [] for i in range(1, 11)}  # Tasks 1 to 10
        operators = [1, 2, 3]
        dir_path = "./output/NO_RESCH/probability_estimation/"

    # Create output destination
    os.makedirs(dir_path, exist_ok=True)

    # Prepare to collect data for the combined plot
    combined_data = []

    for operator in operators:
        # Get data
        path = f'./data/csv/{"RESCH" if rescheduling else "NO_RESCH"}/P0{operator}.csv'
        df = prep_data(path)

        for index, row in df.iterrows():
            tasks = row['opTaskToSave']
            times = row['timetoSave']
            for i in range(len(tasks)):
                task_data[str(tasks[i])].append(times[i])

    file_ln = open(f"{dir_path}/params_ln.txt", "w")
    file_gamma = open(f"{dir_path}/params_gamma.txt", "w")

    # Iterate over task data and create individual plots
    bin_count = 30
    for idx, (key, times) in enumerate(task_data.items()):
        if not times:
            continue
        times = np.array(times)
        combined_data.append((key, times))  # Store for combined plot

        # Create individual plot
        plt.figure(figsize=(8, 4))  # New figure for each task
        plt.hist(times, bins=bin_count, color='skyblue', edgecolor='black', density=True)
        plt.xlabel('Exec time')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Task {key}')

        x_values = np.linspace(min(times), max(times), 1000)

        # Log-Normal MLE
        shape_ln, loc_ln, scale_ln = lognorm.fit(times, floc=0)
        mu_ln = np.log(scale_ln)
        sigma_ln = shape_ln
        line_ln = f'{key} {mu_ln} {sigma_ln}\n'
        file_ln.write(line_ln)

        plt.plot(x_values, lognorm.pdf(x_values, shape_ln, loc_ln, scale_ln), label='Log-Normal MLE')

        # Gamma MLE
        shape_g, loc_g, scale_g = gamma.fit(times, floc=0)
        line_g = f'{key} {shape_g} {scale_g}\n'
        file_gamma.write(line_g)

        plt.plot(x_values, gamma.pdf(x_values, shape_g, loc_g, scale_g), label='Gamma MLE')

        plt.legend()
        plt.savefig(f'{dir_path}/task_{key}.png')
        plt.close()  # Close the figure to free memory

    file_ln.close()
    file_gamma.close()

    # Create combined plot in a grid layout
    num_tasks = len(combined_data)
    cols = 3  # Set the number of columns for the grid
    rows = (num_tasks + cols - 1) // cols  # Calculate rows needed based on the number of tasks

    fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 4), constrained_layout=True)

    for idx, (key, times) in enumerate(combined_data):
        ax = axs[idx // cols, idx % cols]  # Get the correct subplot based on index
        if not times.size:
            continue

        ax.hist(times, bins=bin_count, color='skyblue', edgecolor='black', density=True)
        ax.set_title(f'Histogram of Task {key}')
        ax.set_xlabel('Exec time')
        ax.set_ylabel('Frequency')

        x_values = np.linspace(min(times), max(times), 1000)

        # Log-Normal MLE
        shape_ln, loc_ln, scale_ln = lognorm.fit(times, floc=0)
        ax.plot(x_values, lognorm.pdf(x_values, shape_ln, loc_ln, scale_ln), label='Log-Normal MLE')

        # Gamma MLE
        shape_g, loc_g, scale_g = gamma.fit(times, floc=0)
        ax.plot(x_values, gamma.pdf(x_values, shape_g, loc_g, scale_g), label='Gamma MLE')

        ax.legend()

    # Hide any unused subplots
    for j in range(num_tasks, rows * cols):
        axs[j // cols, j % cols].axis('off')

    plt.savefig(f'{dir_path}/all_tasks_combined.png')
    plt.close()  # Close the figure to free memory

    return task_data


if __name__ == '__main__':
    task_prob_dist(rescheduling=False)
