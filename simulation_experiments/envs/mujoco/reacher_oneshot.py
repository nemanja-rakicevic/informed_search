import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

# 2 link agent
class ReacherOneShotEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        path = os.getcwd()+'/envs/mujoco/assets/'
        mujoco_env.MujocoEnv.__init__(self, path+'reacher_oneshot.xml', 2)

    def _step(self, a):
        # a = a / 180. * np.pi
        a = 10. * a
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 1.
        # self.viewer.cam.lookat[0] += 2.5
        self.viewer.cam.lookat[1] = -0.005
        # self.viewer.cam.lookat[2] += 1.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 90

    def reset_model(self):
        qpos =  self.init_qpos 
        # joint positions
        qpos[0] = -1.
        qpos[1] = 2.
        # ball position
        qpos[2] = 0.
        qpos[3] = 0.
        # qpos[4] = 0.
        # qpos[5] = 1.3

        qvel = self.init_qvel 
        # qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        err = 100*(self.get_body_com("real_target") - self.get_body_com("target"))[:-1]
        return np.concatenate([
            self.model.data.qpos.flat[:],       # joint0, joint1, ballx, bally, targetx, targety
            self.model.data.qvel.flat[:2],      # velocities joint0, joint1
            err                                 # error distance: x,y
        ])


# 5 link agent
class ReacherOneShotEnv_v1(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        path = os.getcwd()+'/envs/mujoco/assets/'
        mujoco_env.MujocoEnv.__init__(self, path+'reacher_oneshot_v1.xml', 2)

    def _step(self, a):
        # a = a / 180. * np.pi
        a = 10. * a
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 1.
        # self.viewer.cam.lookat[0] += 2.5
        self.viewer.cam.lookat[1] = -0.005
        # self.viewer.cam.lookat[2] += 1.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 90

    def reset_model(self):
        qpos =  self.init_qpos 
        # joint positions
        qpos[0] = -1.2
        qpos[1] = 2.5
        qpos[2] = 0.
        qpos[3] = -2.5
        qpos[4] = 1.2
        # ball position
        qpos[5] = 0.
        qpos[6] = 0.
        # qpos[7] = 0.
        # qpos[8] = 1.3

        qvel = self.init_qvel 
        # qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        err = 100*(self.get_body_com("real_target") - self.get_body_com("target"))[:-1]
        return np.concatenate([
            self.model.data.qpos.flat[:],       # joint0, joint1, joint2, joint3, joint4, ballx, bally, targetx, targety
            self.model.data.qvel.flat[:5],      # velocities joint0, joint1, joint2, joint3, joint4
            err                                 # error distance: x,y
        ])