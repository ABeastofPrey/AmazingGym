from gym.envs.registration import register

register(
    id='GridEnv-v0',
    entry_point='custom_envs.grid_env:GridEnv',
)