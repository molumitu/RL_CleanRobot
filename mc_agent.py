from email import policy
import numpy as np
from Env import GridEnv
gamma = 0.9
import copy

def cumulative_return(rewards, gamma):
    future_cumulative_reward = 0.
    cumulative_rewards = np.zeros_like(rewards)
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_rewards[i] = rewards[i] + gamma * future_cumulative_reward
        future_cumulative_reward = cumulative_rewards[i]
    return cumulative_rewards

class StateValue():
    def __init__(self, H, W, seed=1) -> None:
        self.v = np.zeros((H, W))
        self.tem_v = self.v.copy()
        self.v_count = np.zeros_like(self.v, dtype=np.int64)
        self.v_sum = np.zeros_like(self.v)
        self.rng = np.random.default_rng(seed=seed)

    def update(self, obs_trajectory, reward_trajectoty):
        cumulative_rewards = cumulative_return(reward_trajectoty, gamma)
        for i, obs in enumerate(obs_trajectory):
            self.v_count[obs[0], obs[1]] += 1
            self.v_sum[obs[0], obs[1]] += cumulative_rewards[i]
            self.v[obs[0], obs[1]] = self.v_sum[obs[0], obs[1]]/self.v_count[obs[0], obs[1]]
        

    def render(self):
        print(self.v)

    def check_convergence(self):
        pass
        

env = GridEnv()
v_table = StateValue(env.H, env.W)
max_iteration = 1000
episode_num = 100
for i in range(max_iteration):
    for j in range(episode_num):
        obs_trajectory = []
        reward_trajectoty = []
        env.reset()
        init_obs = env.obs.copy()
        obs_trajectory.append(init_obs)
        step = 0
        while(True):
            action = v_table.rng.integers(0, env.action_num)
            next_obs, reward, done, info = env.step(action)
            reward_trajectoty.append(reward)
            if done:
                v_table.update(obs_trajectory, reward_trajectoty)
                break
            obs_trajectory.append(next_obs.copy())
            step += 1
    v_table.render()
    if v_table.check_convergence():
        break
    

    






