import numpy as np
from gym import utils
from gym.envs.mujoco import Walker2dEnv

class WalkerVelEnv(Walker2dEnv):
    def __init__(self):
        self._goal_vel = 0.5
        super(WalkerVelEnv, self).__init__()

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 0.1
        forward_vel = (posafter - posbefore) / self.dt
        forward_reward = - 1.0 * abs(forward_vel - self._goal_vel)

        #reward = ((posafter - posbefore) / self.dt)
        #reward += alive_bonus
        #reward -= 1e-3 * np.square(a).sum()
        ctrl_reward = - 1e-3 * np.square(a).sum()
        reward = forward_reward + ctrl_reward + alive_bonus
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], 
            np.clip(qvel, -10, 10)]).astype(np.float32).flatten().ravel()

    def reset_task(self, task):
        self._goal_vel = task



