import scipy.io
import numpy as np
import pandas as pd
import os
import glob
import ast
import re
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

def clean(x):
    x = x.strip('[]').strip()
    x = re.sub(r'\s+', ' ', x)
    x = x.split()
    return [int(num) for num in x]


def sum_times(arr):
    prefix_sum = []
    current_sum = 0
    for num in arr:
        current_sum += num
        prefix_sum.append(current_sum)
    return prefix_sum


def prep_data(path):
    df = pd.read_csv(path)
    df = df[['opTaskToSave', 'timetoSave', 'mwtoSave']]

    df['timetoSave'] = df['timetoSave'].apply(lambda x: re.sub(r'\s+', ',', x))
    df['timetoSave'] = df['timetoSave'].apply(lambda x: np.squeeze(ast.literal_eval(x)))

    df['opTaskToSave'] = df['opTaskToSave'].apply(lambda x: clean(x))

    df['mwtoSave'] = df['mwtoSave'].apply(lambda x: re.sub(r'\s+', ',', x))
    df['mwtoSave'] = df['mwtoSave'].apply(lambda x: np.squeeze(ast.literal_eval(x)))

    # Apply the prefix sum function to each row in the DataFrame
    df['ExecTimeEnd'] = df['timetoSave'].apply(sum_times)

    return df

def histogram_exec_times_all():
    # Directory containing CSV files
    path = './data/csv/RESCH'

    all_files = glob.glob(os.path.join(path, "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    df = df[['opTaskToSave', 'timetoSave']]

    df['timetoSave'] = df['timetoSave'].apply(lambda x: re.sub(r'\s+', ',', x))
    df['timetoSave'] = df['timetoSave'].apply(lambda x: np.squeeze(ast.literal_eval(x)))

    # Compute the average of the first values across the entire column
    first_values = df['timetoSave'].apply(lambda x: x[0])

    # Plot the histogram of the first values
    plt.hist(first_values, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Exec time')
    plt.ylabel('Frequency')
    plt.title('Histogram of Task 1')
    plt.show()


def histogram_exec_times_op(operator):
    # Get data
    path = './data/csv/RESCH/P0' + str(operator) + '.csv'
    df = prep_data(path)

    # Collect task exec times
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
    for index, row in df.iterrows():
        tasks = row['opTaskToSave']
        times = row['timetoSave']
        for i in range(len(tasks)):
            task_data[str(tasks[i])].append(times[i])

    # Create output destination
    dir_path = f"./output/exec_time_plots/operator_{operator}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create histogram for each task
    bin_count = 10
    for key, values in task_data.items():
        if not values:
            continue
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=bin_count)
        plt.title(f'Task {key}')
        plt.xlabel('Values')
        plt.ylabel('Frequency')

        plt.savefig(f'{dir_path}/task_{key}.png')
        plt.close()


def visualize_2d_time_stress(operator):
    # Get data
    path = './data/csv/RESCH/P0' + str(operator) + '.csv'
    df = prep_data(path)

    # Collect task exec times and stress
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
    for index, row in df.iterrows():
        tasks = row['opTaskToSave']
        times = row['timetoSave']
        stress = row['mwtoSave']
        for i in range(len(tasks)):
            task_data[str(tasks[i])].append((times[i], stress[i]))

    # Create output destination
    dir_path = f"./output/visualize_2d_stress_time/operator_{operator}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create 2D histogram for each task
    bin_count = 4
    for key, values in task_data.items():
        if not values:
            continue

        times, mwtoSave = zip(*values)

        plt.figure(figsize=(8, 6))
        plt.hist2d(times, mwtoSave, bins=(bin_count, bin_count), cmap='viridis')
        plt.colorbar(label='Frequency')
        plt.title(f'Task {key}')
        plt.xlabel('Time')
        plt.ylabel('Stress')

        # Make a grid
        x_edges = np.histogram_bin_edges(times, bins=bin_count)
        y_edges = np.histogram_bin_edges(mwtoSave, bins=bin_count)

        plt.xticks(x_edges, rotation=45)
        plt.yticks(y_edges)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.savefig(f'{dir_path}/task_{key}.png')
        plt.close()


def visualize_strees_over_time(operator):
    # Get data
    path = './data/csv/RESCH/P0' + str(operator) + '.csv'
    df = prep_data(path)
    df = df[['ExecTimeEnd', 'mwtoSave']]


    combined_data = []
    for index, row in df.iterrows():
        combined_data.extend(zip(row['ExecTimeEnd'], row['mwtoSave']))

    filtered_data = [(et, mw) for et, mw in combined_data if mw != 0]
    filtered_exec_times, filtered_mwtosave = zip(*filtered_data)

    # Plotting scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_exec_times, filtered_mwtosave, marker='o', color='blue', alpha=0.8)
    plt.title('Stress over time')
    plt.xlabel('End time of the task')
    plt.ylabel('Stress')


    #############

    # Reshape the data for LinearRegression
    X = np.array(filtered_exec_times).reshape(-1, 1)  # Reshape to a column vector
    y = np.array(filtered_mwtosave)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict y values using the model
    y_pred = model.predict(X)

    # Plot the scatter plot
    plt.scatter(filtered_exec_times, filtered_mwtosave, label='Data Points')

    # Plot the regression line
    plt.plot(filtered_exec_times, y_pred, color='red', label='Linear Regression')
    ############

    plt.legend()
    plt.grid(True)

    # Create output destination
    dir_path = f"./output/visualize_stress_totaltime/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(f'{dir_path}/operator_{operator}.png')


if __name__ == '__main__':
    # standar_exec_times = {'1': 0.5,
    #             '2': 0.667,
    #             '3': 0.333,
    #             '4': 1.0,
    #             '5': 0.5,
    #             '6': 0.5,
    #             '7': 0.333,
    #             '8': 1.0,
    #             '9': 0.667,
    #             '10': 0.5,
    #             '12': 0.667,
    #             '13': 0.5,
    #             '14': 1.0}
    for i in [1, 2, 3, 4, 6, 7]:            # todo: some error generating csv for operator 5
        histogram_exec_times_op(operator = i)
        visualize_2d_time_stress(operator = i)
        visualize_strees_over_time(operator = i)
