
import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import pdb

class BaseStriker(object):
    """ Reacher one shot base class """

    # def _check_collision(self):
    #     model = self.unwrapped.model
    #     self.ball_geom=model.geom_name2id('ball')
    #     self.wall_geoms=[model.geom_name2id(w) for w in model.geom_names \
    #                         if 'side' in w]
    #     self.body_geoms=[model.geom_name2id(w) for w in model.geom_names \
    #                         if 'body' in w or 'fingertip' in w]
    #     for c in range(self.sim.data.ncon):
    #         contact = self.sim.data.contact[c]
    #         if contact.geom1 in self.body_geoms and \
    #            contact.geom2 in self.wall_geoms or \
    #            contact.geom1 in self.wall_geoms and \
    #            contact.geom2 in self.body_geoms:
    #             return True

    def _check_collision(self):
        if self.sim.data.ncon:
            self.contact_cnt += 1
            vec = self.get_body_com("fingertip") - self.get_body_com("ball")
            if self.contact_cnt > 5 and not np.linalg.norm(vec[:2])>0:
                return True

    def reset_model(self):
        self.contact_cnt = 0
        qpos =  self.init_qpos 
        # joint positions
        qpos[:self.num_joints] = self.init_qvals
        # ball position
        qpos[self.num_joints:self.num_joints+2] = np.array([0,0])
        # velocities
        qvel = self.init_qvel 
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, a):
        # a = a / 180. * np.pi
        a = 10. * a
        vec = self.get_body_com("fingertip")-self.get_body_com("ball")
        reward_dist = - np.linalg.norm(vec[:2])
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = self._check_collision()
        return ob, reward, done, dict(reward_dist=reward_dist, 
                                      reward_ctrl=reward_ctrl,
                                      ball_xy=self.get_body_com("ball")[:2])

    def _get_obs(self):
        vec = self.get_body_com("real_target")-self.get_body_com("ball")
        target_dist = 100*np.linalg.norm(vec[:2])
        return np.hstack([
            self.sim.data.qpos.flat[:],       # joint0, .., jointN; ballx, bally; targetx, targety
            self.sim.data.qvel.flat[:-2],     # velocities joint0, .., jointN, ballx, bally
            target_dist                       # error distance: x,y
        ])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 1.
        # self.viewer.cam.lookat[0] += 2.5
        self.viewer.cam.lookat[1] = -0.005
        # self.viewer.cam.lookat[2] += 1.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 90




class Striker2LinkEnv(BaseStriker, mujoco_env.MujocoEnv, utils.EzPickle):
    """ 2 link Reacher one shot agent """
    
    def __init__(self, resolution):
        self.contact_cnt = 0
        utils.EzPickle.__init__(self)
        xml_path = os.path.join(os.getcwd(),
                                'envs/mujoco/assets/reacher_oneshot_2link.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 2)

        self.num_joints = 2
        self.parameter_list = np.array([np.linspace(-1.57, 1.57, resolution),
                                        np.linspace(-3.14, 3.14, resolution)])
        self.init_qvals = np.array([-1, 2])



class Striker5LinkEnv(BaseStriker, mujoco_env.MujocoEnv, utils.EzPickle):
    """ 5 link Reacher one shot agent """

    def __init__(self, resolution):
        self.contact_cnt = 0
        utils.EzPickle.__init__(self)
        xml_path = os.path.join(os.getcwd(),
                                'envs/mujoco/assets/reacher_oneshot_5link.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 2)

        self.num_joints = 5
        self.parameter_list = np.array([np.linspace(-1.57, 1.57, resolution),
                                        np.linspace(-3.14, 3.14, resolution),
                                        np.linspace(-3.14, 3.14, resolution),
                                        np.linspace(-3.14, 3.14, resolution),
                                        np.linspace(-3.14, 3.14, resolution)])
        self.init_qvals = np.array([-1.2, 2.5, 0., -2.5, 1.2])
