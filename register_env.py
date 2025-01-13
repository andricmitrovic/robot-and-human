from gymnasium.envs.registration import register

register(
    id='CollaborationEnv-v0',
    entry_point='env:CollaborationEnv',
    kwargs={'operator': 'avg'}
)