import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define constants for task categories
HUMAN_TASKS = np.array([1, 2, 3, 4, 5])  # Example tasks
ROBOT_TASKS = np.array([6, 7, 8, 9, 10])  # Example tasks
COMMON_TASKS = np.array([11, 12, 13])  # Example tasks
SAMPLE_SIZE = 100  # Number of actions to sample


# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)  # Combined input
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Output is a single value

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output the value


def sample_valid_action(state):
    remaining_tasks = state[3]
    available_human_tasks = [task for task in remaining_tasks if task in HUMAN_TASKS]
    available_robot_tasks = [task for task in remaining_tasks if task in ROBOT_TASKS]
    available_common_tasks = [task for task in remaining_tasks if task in COMMON_TASKS]

    # Randomly assign common tasks
    np.random.shuffle(available_common_tasks)
    split_index = np.random.randint(0, len(available_common_tasks) + 1)
    common_for_human = available_common_tasks[:split_index]
    common_for_robot = available_common_tasks[split_index:]

    # Create final task schedules
    human_schedule = available_human_tasks + common_for_human
    robot_schedule = available_robot_tasks + common_for_robot

    return (human_schedule, robot_schedule)


def encode_action(action):
    # Encode action for human and robot separately
    encoded_action_human = np.zeros(13)  # Adjust size based on your task count
    encoded_action_robot = np.zeros(13)

    for task in action[0]:  # Human tasks
        encoded_action_human[task - 1] = 1
    for task in action[1]:  # Robot tasks
        encoded_action_robot[task - 1] = 1

    # Concatenate encoded actions for both human and robot
    return np.concatenate((encoded_action_human, encoded_action_robot))


def select_best_action(value_network, state):
    valid_actions = []

    # Sample valid actions
    for _ in range(SAMPLE_SIZE):
        action = sample_valid_action(state)
        valid_actions.append(action)

    # Evaluate each action using the Value Network
    action_values = []

    for action in valid_actions:
        # Prepare state and action tensors
        state_tensor = torch.FloatTensor(np.concatenate(([state[0], state[1], state[2]], state[3]))).unsqueeze(0)
        action_tensor = torch.FloatTensor(encode_action(action)).unsqueeze(0)
        value = value_network(state_tensor, action_tensor).item()
        action_values.append(value)

    # Select the best action based on the highest value
    best_action_index = np.argmax(action_values)
    best_action = valid_actions[best_action_index]

    return best_action


if __name__ == '__main__':
    state_dim = 23  # 3 state components + 20 remaining tasks
    action_dim = 13 * 2  # 13 tasks for human and 13 tasks for robot
    value_network = ValueNetwork(state_dim, action_dim)
    optimizer = optim.Adam(value_network.parameters(), lr=0.001)

    # Example usage
    state = (0, 0, 0, np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))  # Example state
    best_action = select_best_action(value_network, state)
    print("Best action selected:", best_action)
