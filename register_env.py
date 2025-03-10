from gymnasium.envs.registration import register

# Original environment
register(
    id='CollaborationEnv-v0',
    entry_point='env:CollaborationEnv',
    kwargs={'operator': 'avg'}
)

# Improved original environment
register(
    id='CollaborationEnv-v1',
    entry_point='env:CollaborationEnv_V1',
    kwargs={'operator': 'avg'}
)

# One action environment
# register(
#     id='CollaborationEnv-v1',
#     entry_point='env_v1:CollaborationEnvV1',
#     kwargs={'operator': 'sum'}
# )