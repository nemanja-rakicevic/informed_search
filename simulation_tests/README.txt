
# Adding the ReacherOneShotEnv to gym

1) Add the environment files to the gym directories
$ cp reacher_oneshot.py ~/src/gym/gym/envs/mujoco/
$ cp reacher_oneshot.xml ~/src/gym/gym/envs/mujoco/assets/

2) Add the entry to the  ~/src/gym/gym/envs/__init__.py
---
register(
    id='ReacherOneShot-v0',
    entry_point='gym.envs.mujoco:ReacherOneShotEnv',
    max_episode_steps=50,
    reward_threshold=0,
)
---