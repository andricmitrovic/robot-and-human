import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from utils import prep_data


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
    dir_path = f"./output/exec_time_frequency/operator_{operator}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create histogram for each task
    bin_count = 10
    for key, values in task_data.items():
        if not values:
            continue
        plt.figure(figsize=(8, 6))
        hist_values, bin_edges, _ = plt.hist(values, bins=bin_count, color='skyblue', edgecolor='black')
        plt.title(f'Task {key}')
        plt.xlabel('Execution time')
        plt.ylabel('Frequency')
        plt.xticks(bin_edges)
        plt.savefig(f'{dir_path}/task_{key}.png')
        plt.close()


def visualize_3d_time_stress(operator):
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
    dir_path = f"./output/exec_time_stress_frequency/operator_{operator}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create 2D histogram for each task
    bin_count = 4
    for key, values in task_data.items():
        if not values:
            continue

        times, mwtoSave = zip(*values)
        times = np.array(times)
        mwtoSave = np.array(mwtoSave)

        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(times, mwtoSave, bins=(bin_count, bin_count))

        # Create meshgrid for plotting
        x_data, y_data = np.meshgrid(np.arange(hist.shape[1]), np.arange(hist.shape[0]))

        # Flatten data
        x_data = x_data.flatten()
        y_data = y_data.flatten()
        z_data = hist.flatten()

        # Plot 3D histogram
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data, color=plt.cm.magma(z_data/np.max(z_data)))

        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Stress')
        ax.set_zlabel('Frequency')
        ax.set_title('3D Histogram')

        # Set tick positions and labels
        ax.set_xticks(np.arange(bin_count))
        ax.set_yticks(np.arange(bin_count))
        ax.set_xticklabels(np.round(x_edges[:-1], 2))  # Adjust to match the number of bins
        ax.set_yticklabels(np.round(y_edges[:-1], 2))  # Adjust to match the number of bins

        plt.savefig(f'{dir_path}/task_{key}.png')
        plt.close()


def visualize_strees_over_time(operator):
    # Get data
    path = './data/csv/RESCH/P0' + str(operator) + '.csv'
    df = prep_data(path)
    df = df[['timetoSave', 'mwtoSave']]

    stress_history = []
    times_history = []
    cycle_end_ids = []
    curr_time = 0
    for index, row in df.iterrows():
        times = row['timetoSave']
        stress = row['mwtoSave']
        for i in range(len(times)):
            curr_time += times[i]
            times_history.append(curr_time)
            stress_history.append(stress[i])
        cycle_end_ids.append(len(times_history))

    # Plotting scatter
    plt.figure(figsize=(20, 5))
    plt.scatter(times_history, stress_history, marker='o', color='blue', alpha=0.8)
    plt.title('Stress over time')
    plt.xlabel('End time of the task')
    plt.ylabel('Stress')

    # Plot cycle separators
    for point_id in cycle_end_ids:
        point_x = times_history[point_id-1]
        plt.axvline(x=point_x, color='green', linestyle='--', linewidth=1)
    plt.axvline(x=float('inf'), color='green', linestyle='--', linewidth=1, label='End of cycle')

    #############
    # Perform Polynomial Regression
    degree = 3  # Degree of polynomial
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(np.array(times_history).reshape(-1, 1))

    # Fit polynomial regression model
    model = LinearRegression()
    model.fit(X_poly, stress_history)

    # Predict y values using the model
    y_pred = model.predict(X_poly)

    # Sort the values for plotting
    sorted_zip = sorted(zip(times_history, y_pred))
    times_history, y_pred = zip(*sorted_zip)

    # Plot the regression line
    plt.plot(times_history, y_pred, color='red', label=f'Polynomial Regression (Degree {degree})')
    ############

    # Perform Polynomial Regression for each cycle
    degree = 1  # Degree of polynomial
    for i in range(len(cycle_end_ids)):
        start_idx = 0 if i == 0 else cycle_end_ids[i - 1]
        end_idx = cycle_end_ids[i]
        cycle_times = times_history[start_idx:end_idx]
        cycle_stress = stress_history[start_idx:end_idx]

        polynomial_features = PolynomialFeatures(degree=degree)
        X_poly = polynomial_features.fit_transform(np.array(cycle_times).reshape(-1, 1))

        # Fit polynomial regression model
        model = LinearRegression()
        model.fit(X_poly, cycle_stress)

        # Predict y values using the model
        y_pred = model.predict(X_poly)

        # Plot the regression line for the current cycle
        if i == 0:
            plt.plot(cycle_times, y_pred, color='orange', label = f'Cycle Polynomial Regression (Degree {degree})')
        else:
            plt.plot(cycle_times, y_pred, color='orange')
    ############

    plt.legend()
    plt.grid(True)

    # Create output destination
    dir_path = f"./output/stress_over_time/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(f'{dir_path}/operator_{operator}.png')
    plt.close()

def exec_time_per_cycle(operator):
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
            task_data[str(tasks[i])].append((times[i], index))

    # Create output destination
    dir_path = f"./output/exec_time_per_cycle/operator_{operator}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for key, values in task_data.items():
        if not values:
            continue

        times, cycle = zip(*values)
        plt.figure(figsize=(8, 6))
        plt.plot(cycle, times)
        plt.scatter(cycle, times, color='red')
        plt.title(f'Task {key}')
        plt.xlabel('Cycle')
        plt.ylabel('Exec time')
        plt.xticks(cycle)

        plt.savefig(f'{dir_path}/task_{key}.png')
        plt.close()


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
        visualize_3d_time_stress(operator = i)
        visualize_strees_over_time(operator = i)
        exec_time_per_cycle(operator = i)


    # state = {tasks done and not done}, which one is human doing, since when
    # actions = robot performs
