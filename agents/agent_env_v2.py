import gymnasium as gym
import register_env
import numpy as np
from itertools import product
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

np.random.seed(42)

human_tasks = [1, 2, 3, 4, 5, 6, 10]
robot_tasks = [11, 15, 16, 17, 18, 19, 20]
common_tasks = [7, 8, 9, 12, 13, 14]


def generate_all_schedules():
    assignments = []
    for combo in product([0, 1], repeat=len(common_tasks)):
        human = human_tasks.copy()
        robot = robot_tasks.copy()
        label = []
        for i, assign_to in enumerate(combo):
            if assign_to == 0:
                human.append(common_tasks[i])
                label.append('H')
            else:
                robot.append(common_tasks[i])
                label.append('R')
        label_str = "(" + ",".join(label) + ")"
        assignments.append((human, robot, label_str))
    return assignments


def save_top_5_boxplot(reward_history, top_5, filename="../output/plots/top_5_rewards_boxplot.png"):
    data = []
    for label, _ in top_5:
        for reward in reward_history[label]:
            data.append({'Schedule': label, 'Reward': reward})

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        x='Schedule',
        y='Reward',
        hue='Schedule',
        data=df,
        palette="Set2",
        showmeans=True,
        meanline=True,
        meanprops={"color": "black", "linestyle": "--", "linewidth": 2}
    )

    plt.title("Reward Distributions for Top 5 Schedules (Mean as Line)")
    plt.ylabel("Reward")
    plt.xlabel("Schedule Label")

    # Remove redundant legend
    if ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Boxplot saved as: {filename}")


all_schedules = generate_all_schedules()
num_episodes = 1000
env = gym.make('CollaborationEnv-v2')
reward_history = {}

for human_schedule, robot_schedule, label in tqdm(all_schedules):
    reward_history[label] = []
    for episode in range(num_episodes):
        env.reset()
        action = (human_schedule, robot_schedule)
        observation, reward, terminated, truncated, info = env.step(action)
        reward_history[label].append(reward)
env.close()

######
# Step 1: Compute average rewards for each label
average_rewards = {
    label: sum(rewards) / len(rewards)
    for label, rewards in reward_history.items()
}

# Step 2: Sort labels by average reward, descending
sorted_labels = sorted(
    average_rewards.items(),
    key=lambda x: x[1],
    reverse=True
)

# Step 3: Get top 5 labels
top_5 = sorted_labels[:5]

# Print results
print("Top 5 Labels with Best Average Reward:")
for rank, (label, avg_reward) in enumerate(top_5, 1):
    print(f"{rank}. Label: {label}, Avg Reward: {avg_reward:.2f}")


save_top_5_boxplot(reward_history, top_5)



