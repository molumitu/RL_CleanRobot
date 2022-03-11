import gym
import numpy as np

class GridEnv(gym.Env):
    def __init__(self, seed=0) -> None:
        self.H = 4
        self.W = 4
        self.move_option = np.array([[0,1], [-1,0], [0,-1], [1,0]]) #top, left, down, right
        self.action_num = 4
        self.rng = np.random.default_rng(seed=seed)
        self.obs = self.reset()

    def reset(self):
        init_robot_h, init_robot_w = self._init_robot_pos()
        self.obs = np.array([init_robot_h, init_robot_w])

        return self.obs.copy()

    def step(self, action):
        P_action_random =  self.rng.uniform(0,1)
        if P_action_random < 0.1:
            action += 1
        elif P_action_random > 0.9:
            action -= 1
        action %= self.action_num

        self.obs += self.move_option[action]
        self.obs = np.clip(self.obs, a_min=[0, 0], a_max=[self.H-1, self.W-1])

        done = self._judge_done()
        reward = self._cal_reward(done)
        
        info = {}
        return self.obs.copy(), reward, done, info
    
    def render(self):
        pass

    def _init_robot_pos(self):
        index = self.rng.integers(self.H*self.W-1)
        h = index // self.W
        w = index % self.W
        return h, w

    def _cal_reward(self, done):
        return -1. + 10. * done

    def _judge_done(self):
        if (self.obs == np.array([self.H-1, self.W-1])).all():
            return True
        else:
            return False

        

    
