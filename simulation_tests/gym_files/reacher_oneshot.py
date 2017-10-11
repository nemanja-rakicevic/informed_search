import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np


class ReacherOneShotEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_oneshot.xml', 2)

    def _step(self, a):
        # a = a / 180. * np.pi
        a = 10.*a
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        # print("\n")
        # print(dir(self.model))
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * .9
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 90

    def reset_model(self):
        qpos =  0*self.init_qpos 
        qpos[0] = -1.
        qpos[1] = 2.
        qpos[2] = 0.
        qpos[3] = 0.

        qvel = self.init_qvel 
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:],       # joint0, joint1, ballx, bally
            self.model.data.qvel.flat[:2],      # velocities -//-
            self.get_body_com("fingertip") - self.get_body_com("target"),    # distance: x,y,z
        ])
