import numpy as np
from gym import utils
from gym.envs.mujoco import AntEnv


class AntVelEnv(AntEnv):
    def __init__(self):
        self._goal_vel = 0.25
        super(AntVelEnv, self).__init__()

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        fwd_vel = (xposafter - xposbefore) / self.dt
        fwd_reward = - 2.0 * abs(fwd_vel - self._goal_vel)

        ctrl_cost = - 1e-2 * np.square(a).sum()
        contact_cost = - 1e-4 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.1

        reward = fwd_reward + ctrl_cost + contact_cost + survive_reward
        #reward = fwd_reward + survive_reward
        #print('foward reward: %.2f'%fwd_reward)

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=fwd_reward,
            reward_ctrl=ctrl_cost,
            reward_contact=contact_cost,
            reward_survive=survive_reward,
            velocity=fwd_vel)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ]).flatten().astype(np.float32)

    def reset_task(self, task):
        self._goal_vel = task

    def sample_task(self, num_tasks=1):
        tasks = np.random.uniform(0.0, 0.5, size=(num_tasks, 2))
        return tasks


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5




class AntNaviEnv(AntEnv):
    def __init__(self):
        self._goal = [1.0, 1.0]
        super(AntNaviEnv, self).__init__()

    def step(self, a):

        a = np.clip(a, -1.0, 1.0)
        #xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        pos_x = self.get_body_com("torso")[0]
        pos_y = self.get_body_com("torso")[1]

        ctrl_cost = - 1e-2 * np.square(a).sum()
        contact_cost = - 1e-4 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.2

        x = pos_x - self._goal[0]
        y = pos_y - self._goal[1]
        reward_dist = - np.sqrt(x ** 2 + y ** 2)
        done_dist = (np.abs(x) < 0.01) and (np.abs(y) < 0.01)

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done_fall = not notdone

        #reward_fall = -10 if done_fall else 0
        reward = reward_dist + survive_reward + ctrl_cost + contact_cost

        done = done_dist or done_fall
        ob = self._get_obs()
        return ob, reward, done, {'xy': np.array([pos_x, pos_y])}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ]).flatten().astype(np.float32)

    def reset_task(self, task):
        self._goal = list(task)

    def reset_model(self):
        qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
