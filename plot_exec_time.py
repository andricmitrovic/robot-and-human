import scipy.io
import numpy as np
import pandas as pd
import os
import glob
import ast
import re
from matplotlib import pyplot as plt

def clean(x):
    x = x.strip('[]').strip()
    x = re.sub(r'\s+', ' ', x)
    x = x.split()
    return [int(num) for num in x]

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
    path = './data/csv/RESCH/P0' + str(operator) + '.csv'
    df = pd.read_csv(path)
    df = df[['opTaskToSave', 'timetoSave']]

    df['timetoSave'] = df['timetoSave'].apply(lambda x: re.sub(r'\s+', ',', x))
    df['timetoSave'] = df['timetoSave'].apply(lambda x: np.squeeze(ast.literal_eval(x)))

    df['opTaskToSave'] = df['opTaskToSave'].apply(lambda x: clean(x))

    # print(df)
    tasks_times = {'1': [],
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
            tasks_times[str(tasks[i])].append(times[i])

    dir_path = f"./output/exec_time_plots/operator_{operator}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for key, values in tasks_times.items():
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=10)
        plt.title(f'Task {key}')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'{dir_path}/task{key}.png')
        plt.close()

if __name__ == '__main__':
    # std_exec = {'1': 0.5,
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

