
from gym.envs.registration import register


register(
    id='Striker2LinkEnv-v0',
    entry_point='envs.mujoco.striker_oneshot:Striker2LinkEnv',
    max_episode_steps=100,
    reward_threshold=0,
)

register(
    id='Striker5LinkEnv-v0',
    entry_point='envs.mujoco.striker_oneshot:Striker5LinkEnv',
    max_episode_steps=100,
    reward_threshold=0,
)
