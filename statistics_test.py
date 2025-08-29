import numpy as np
from scipy import stats

operator_type = 'avg'

# Example: arrays of rewards from simulations
dqn_rewards = np.load(f"./output/dqn_rewards_{operator_type}.npy")       # e.g. 10,000 rewards
static_rewards = np.load(f"./output/static_rewards_{operator_type}.npy") # e.g. 10,000 rewards for best policy

# Independent two-sample t-test
t_stat, p_val = stats.ttest_ind(dqn_rewards, static_rewards, equal_var=False)

print("t-statistic:", t_stat)
print("p-value:", p_val)
