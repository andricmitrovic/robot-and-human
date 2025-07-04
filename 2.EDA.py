import numpy as np
import os

import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from utils import prep_data
import joblib
from scipy.stats import linregress

robotExecTime = {7: 0.372,
                 8: 1.1,
                 9: 0.685,
                 11: 0.47,
                 12: 0.489,
                 13: 0.271,
                 14: 1.1,
                 15: 0.62,
                 16: 0.333,
                 17: 0.23,
                 18: 0.878,
                 19: 0.809,
                 20: 0.711
                 }

def histogram_exec_times_op(operator, normalize = True, rescheduling = True):
    # Get data
    if rescheduling:
        path = './data/csv/RESCH/P0' + str(operator) + '.csv'
    else:
        path = './data/csv/NO_RESCH/P0' + str(operator) + '.csv'
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
    if rescheduling:
        dir_path = f"./output/RESCH/exec_time_frequency/operator_{operator}"
    else:
        dir_path = f"./output/NO_RESCH/exec_time_frequency/operator_{operator}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create histogram for each task
    bin_count = 10
    bin_width = 0.5
    half_width = bin_width * (bin_count // 2)
    bins = np.linspace(-half_width, half_width, bin_count + 1)
    for key, values in task_data.items():
        if not values:
            continue
        times = np.array(values)
        if normalize and len(times) > 1:
            mean = np.mean(times)
            std = np.std(times)
            times = (times-mean) / std

        plt.figure(figsize=(8, 6))
        hist_values, bin_edges, _ = plt.hist(times, bins=bins, color='skyblue', edgecolor='black')
        plt.title(f'Task {key}')
        plt.xlabel('Normalized execution time')
        plt.ylabel('Frequency')
        plt.xticks(bin_edges)
        # bin_width = bin_edges[1] - bin_edges[0]
        # plt.xticks(np.arange(-bin_width * bin_count / 2, bin_width * (bin_count / 2 + 1), bin_width))
        plt.savefig(f'{dir_path}/task_{key}.png')
        plt.close()


def box_plot_exec_time_vs_robot():
    operators = [1, 2, 3, 4, 6, 7]
    path = './data/csv/RESCH/P0'

    task_data = {
            '7': [],
            '8': [],
            '9': [],
            '12': [],
            '13': [],
            '14': []}

    for operator in operators:
        df = prep_data(path + str(operator) + '.csv')

        # Collect task exec times
        for index, row in df.iterrows():
            tasks = row['opTaskToSave']
            times = row['timetoSave']
            for i in range(len(tasks)):
                task_id = str(tasks[i])
                if task_id in task_data.keys():
                    task_data[task_id].append(times[i])

    dir_path = f"./output/plots/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # print(task_data)

    task_ids = list(task_data.keys())
    data = [task_data[task] for task in task_ids]

    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(data, patch_artist=True)

    # Style the boxes
    for patch in box['boxes']:
        patch.set_facecolor('#add8e6')

    ax.set_xticks(range(1, len(task_ids) + 1))
    ax.set_xticklabels(task_ids)
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Operator Execution Time per Task")

    # Add horizontal lines for robot execution times
    for i, task in enumerate(task_ids):
        robot_time = robotExecTime.get(int(task))
        if robot_time is not None:
            # Draw a short horizontal line across the boxplot
            ax.hlines(robot_time, i + 0.8, i + 1.2, colors='red', linewidth=2, label='Robot Exec Time' if i == 0 else "")

    ax.legend()
    ax.grid(True)

    dir_path = f"./output/plots/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # plt.show()
    plt.savefig(os.path.join(dir_path, "boxplot_exec_vs_robot.png"))
    plt.close()

    return


def visualize_3d_time_stress(operator, normalize = True, rescheduling = True):
    # Get data
    if rescheduling:
        path = './data/csv/RESCH/P0' + str(operator) + '.csv'
    else:
        path = './data/csv/NO_RESCH/P0' + str(operator) + '.csv'
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
    if rescheduling:
        dir_path = f"./output/RESCH/exec_time_stress_frequency/operator_{operator}"
    else:
        dir_path = f"./output/NO_RESCH/exec_time_stress_frequency/operator_{operator}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create 2D histogram for each task
    bin_count = 4
    bin_width = 1
    half_width = bin_width * (bin_count // 2)
    bins = np.linspace(-half_width, half_width, bin_count + 1)
    for key, values in task_data.items():
        if not values:
            continue

        times, mwtoSave = zip(*values)
        times = np.array(times)
        if normalize and len(times) > 1:
            mean = np.mean(times)
            std = np.std(times)
            times = (times - mean) / std

        mwtoSave = np.array(mwtoSave)

        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(times, mwtoSave, bins=(bins, bin_count))

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
        ax.set_xlabel('Normalized Exec Time')
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


def visualize_strees_over_time(operator, rescheduling = True):
    # Get data
    if rescheduling:
        path = './data/csv/RESCH/P0' + str(operator) + '.csv'
    else:
        path = './data/csv/NO_RESCH/P0' + str(operator) + '.csv'
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
    # RANSAC Regression
    ransac = RANSACRegressor()
    X = np.array(times_history).reshape(-1, 1)
    y = np.array(stress_history)
    ransac.fit(X, y)

    # Predict y values using the RANSAC model
    y_pred_ransac = ransac.predict(X)

    # Sort the values for plotting
    sorted_zip = sorted(zip(times_history, y_pred_ransac))
    times_history, y_pred_ransac = zip(*sorted_zip)

    # Plot the RANSAC regression line
    plt.plot(times_history, y_pred_ransac, color='purple', label='RANSAC Regression', linewidth=2)
    ############

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
    degree = 3  # Degree of polynomial
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
            plt.plot(cycle_times, y_pred, color='orange', label = f'Experiment Polynomial Regression (Degree {degree})')
        else:
            plt.plot(cycle_times, y_pred, color='orange')
    ############

    plt.legend()
    plt.grid(True)
    plt.xlim(0, max(times_history))
    plt.tight_layout()


    # Create output destination
    if rescheduling:
        dir_path = f"./output/RESCH/stress_over_time/"
    else:
        dir_path = f"./output/NO_RESCH/stress_over_time/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(f'{dir_path}/operator_{operator}.png')
    plt.close()


def exec_time_per_cycle(operator, normalize = True, rescheduling = True):
    # Get data
    if rescheduling:
        path = './data/csv/RESCH/P0' + str(operator) + '.csv'
    else:
        path = './data/csv/NO_RESCH/P0' + str(operator) + '.csv'
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
    if rescheduling:
        dir_path = f"./output/RESCH/exec_time_per_cycle/operator_{operator}"
    else:
        dir_path = f"./output/NO_RESCH/exec_time_per_cycle/operator_{operator}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for key, values in task_data.items():
        if not values:
            continue

        times, cycle = zip(*values)
        times = np.array(times)
        if normalize and len(times) > 1:
            mean = np.mean(times)
            std = np.std(times)
            times = (times - mean) / std
        plt.figure(figsize=(8, 6))
        plt.plot(cycle, times)
        plt.scatter(cycle, times, color='red')
        plt.title(f'Task {key}')
        plt.xlabel('Cycle')
        plt.ylabel('Normalized exec time')
        plt.xticks(cycle)

        plt.savefig(f'{dir_path}/task_{key}.png')
        plt.close()


def hist_tasks():
    operators = [1, 2, 3, 4, 6, 7]
    path = './data/csv/RESCH/P0'

    task_counter = {x:{} for x in operators}
    for op in operators:
        df = prep_data(path + str(op) + '.csv')
        df = df[['opTaskToSave']]
        for _, row in df.iterrows():
            for i in row['opTaskToSave']:
                if i in task_counter[op].keys():
                    task_counter[op][i] += 1
                else:
                    task_counter[op][i] = 1

    # print(task_counter[1])
    #
    # keys = list(task_counter[1].keys())
    # values = list(task_counter[1].values())
    #
    # plt.figure(figsize=(10, 6))
    # plt.bar(keys, values, color='skyblue', edgecolor='black')
    #
    # plt.xlabel('Task')
    # plt.ylabel('Count')
    # plt.title(f'Operator {}')
    # plt.xticks(keys)  # Show all keys on x-axis
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    #
    # # Show the plot
    # plt.tight_layout()
    # plt.show()

    # Create combined plot in a grid layout
    num_op = len(operators)
    cols = 2
    rows = (num_op + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 4), constrained_layout=True)

    for idx, op in enumerate(operators):
        ax = axs[idx // cols, idx % cols]

        keys = list(task_counter[op].keys())
        values = list(task_counter[op].values())

        ax.bar(keys, values, color='skyblue', edgecolor='black')
        ax.set_xticks(range(1, 21))
        ax.set_title(f'Operator {op}')
        ax.set_xlabel('Task')
        ax.set_ylabel('Count')

    # Hide any unused subplots
    for j in range(num_op, rows * cols):
        axs[j // cols, j % cols].axis('off')

    plt.savefig(f'./output/RESCH/task_count.png')
    plt.close()


def extract_stress_summary():
    results = {}
    for operator in [1, 2, 3, 4, 6, 7]:
        key = f'operator {operator}'
        results[key] = {}

        path = './data/csv/RESCH/P0' + str(operator) + '.csv'
        df = prep_data(path)
        df = df[['timetoSave', 'mwtoSave']]

        cycle_stress = []
        cycle_times = []
        for index, row in df.iterrows():
            # Append stress
            cycle_stress.append(row['mwtoSave'])
            # Append current times
            exec_times = row['timetoSave']
            current_time = 0
            times_array = []
            for i in range(len(exec_times)):
                current_time += exec_times[i]
                times_array.append(current_time)
            cycle_times.append(times_array)

        num_cycles = len(cycle_stress)
        for i in range(num_cycles):
            stress = cycle_stress[i]
            time = cycle_times[i]

            # Basic stats
            mean_val = np.mean(stress)
            max_val = np.max(stress)
            min_val = np.min(stress)
            std_val = np.std(stress)
            range_val = max_val - min_val
            first_val = stress[0]
            last_val = stress[-1]

            threshold = mean_val + std_val

            # Trend (slope)
            slope, _, _, _, _ = linregress(time, stress)

            # Rate of change
            rate_of_change = np.mean(np.abs(np.diff(stress)))

            # Peaks
            peaks, _ = find_peaks(stress)
            num_peaks = len(peaks)

            # Duration above threshold
            time_above_threshold = 0.0
            for k in range(len(stress) - 1):
                if stress[k] > threshold:
                    delta_t = time[k + 1] - time[k]
                    time_above_threshold += delta_t

            # Append results
            results[key][i] = {
                                    "mean": mean_val,
                                    "max": max_val,
                                    "min": min_val,
                                    "std": std_val,
                                    "range": range_val,
                                    "first value": first_val,
                                    "last value": last_val,
                                    "trend slope": slope,
                                    "rate of change": rate_of_change,
                                    "num peaks": num_peaks,
                                    "threshold": threshold,
                                    "duration above threshold": time_above_threshold
                                }

    # Flatten into rows
    rows = []
    for operator, cycles in results.items():
        for cycle_id, stats in cycles.items():
            row = {'operator': operator, 'cycle id': cycle_id, **stats}
            rows.append(row)

    df = pd.DataFrame(rows)
    df.round(2).to_csv("./output/RESCH/stress_metrics_all.csv")
    df.round(2).to_latex('./output/latex/stress_metrics_all.tex', index=False, float_format="%.2f")

    df = df.drop(columns='cycle id')
    df = df.groupby('operator').mean().reset_index()
    df.round(2).to_csv("./output/RESCH/stress_metrics.csv")
    df.round(2).to_latex('./output/latex/stress_metrics.tex', index=False, float_format="%.2f")

    return


if __name__ == '__main__':
    os.makedirs('./output/NO_RESCH', exist_ok=True)
    os.makedirs('./output/RESCH', exist_ok=True)
    # for i in [1, 2, 3, 4, 6, 7]:            # todo: some error generating csv for operator 5
        # histogram_exec_times_op(operator=i)
        # visualize_3d_time_stress(operator=i)
        # visualize_strees_over_time(operator=i)
        # exec_time_per_cycle(operator=i)

    # for i in [1, 2, 3]:
    #     histogram_exec_times_op(operator=i, rescheduling=False)
    #     visualize_3d_time_stress(operator=i, rescheduling=False)
    #     visualize_strees_over_time(operator=i, rescheduling=False)
    #     exec_time_per_cycle(operator=i, rescheduling=False)



    # hist_tasks()

    # box_plot_exec_time_vs_robot()

    extract_stress_summary()

