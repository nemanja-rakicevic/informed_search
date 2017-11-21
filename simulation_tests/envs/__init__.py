from gym.envs.registration import register

# 2 link
register(
    id='ReacherOneShot-v0',
    entry_point='envs.mujoco.reacher_oneshot:ReacherOneShotEnv',
    max_episode_steps=50,
    reward_threshold=0,
)

# 5 link
register(
    id='ReacherOneShot-v1',
    entry_point='envs.mujoco.reacher_oneshot:ReacherOneShotEnv_v1',
    max_episode_steps=50,
    reward_threshold=0,
)

# from gym.envs.registration import register

# # discrete environments
# register(id='LastMoment-v0', entry_point='envs.discrete.last_moment_env:LastMomentEnv', max_episode_steps=3)
# register(id='QueueOfCars-v0', entry_point='envs.discrete.queue_of_cars_env:QueueOfCarsEnv', max_episode_steps=15)
# register(id='GridWorld-v0', entry_point='envs.discrete.grid_world_env:GridWorldEnv', max_episode_steps=3)

# # MuJoCo environments
# register(id='InfiniteCubePusher-v0', entry_point='envs.mujoco.infinite_cube_pusher_env:InfiniteCubePusherEnv', max_episode_steps=1000)