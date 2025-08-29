from gymnasium.envs.registration import register

# Original environment
register(
    id='CollaborationEnv-v0',
    entry_point='env_v0:CollaborationEnv_v0',
    kwargs={'operator': 'avg'}
)

# Improved original environment
register(
    id='CollaborationEnv-v1',
    entry_point='env_v1:CollaborationEnv_V1',
    kwargs={'operator': 'avg'}
)

# Full schedule executed in one step
register(
    id='CollaborationEnv-v2',
    entry_point='env_v2:CollaborationEnv_V2',
    kwargs={'operator': 'avg'}
)

# Action call every time human/robot finishes
register(
    id='CollaborationEnv-v3',
    entry_point='env_v3:CollaborationEnv_V3',
    kwargs={'operator': 'avg'}
)

# Action call every time human/robot finishes
register(
    id='CollaborationEnv-v4',
    entry_point='env_v4:CollaborationEnv_V4',
    kwargs={'operator': 'avg'}
)

# Action call every time human/robot finishes
register(
    id='CollaborationEnv-v5',
    entry_point='env_v5:CollaborationEnv_V5',
    kwargs={'operator': 'avg'}
)

# Action call every time human/robot finishes
register(
    id='CollaborationEnv-v6',
    entry_point='env_v6:CollaborationEnv_V6',
    kwargs={'operator': 'avg'}
)

# Simplified with only common task dynamic reschedule
register(
    id='CollaborationEnv-v7',
    entry_point='env_v7:CollaborationEnv_V7',
    kwargs={'operator': 'avg'}
)
