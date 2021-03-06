import numpy as np
from Env import GridEnv
import copy

def cumulative_return(rewards, gamma):
    future_cumulative_reward = 0.
    cumulative_rewards = np.zeros_like(rewards)
    for i in range(len(rewards) - 1, -1, -1):
        cumulative_rewards[i] = rewards[i] + gamma * future_cumulative_reward
        future_cumulative_reward = cumulative_rewards[i]
    return cumulative_rewards

class StateValue():
    def __init__(self, H, W, seed=0) -> None:
        self.update_times = 0
        self.v = np.zeros((H, W))
        self.tem_v = self.v.copy()
        self.v_count = np.zeros_like(self.v, dtype=np.int64)
        self.v_sum = np.zeros_like(self.v)
        self.rng = np.random.default_rng(seed=seed)

    def update(self, obs_trajectory, reward_trajectoty):
        self.update_times += 1
        cumulative_rewards = cumulative_return(reward_trajectoty, gamma)
        for i, obs in enumerate(obs_trajectory):
            self.v_count[obs[0], obs[1]] += 1
            self.v_sum[obs[0], obs[1]] += cumulative_rewards[i]
            self.v[obs[0], obs[1]] = self.v_sum[obs[0], obs[1]]/self.v_count[obs[0], obs[1]]
        

    def render(self):
        print(self.v)

    def check_convergence(self):
        done = np.allclose(self.v, self.tem_v, atol=0.1)
        self.tem_v = self.v.copy()
        return done
        
gamma = 0.9
env = GridEnv()
v_table = StateValue(env.H, env.W)
max_iteration = 2000
episode_num = 1000
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
    if v_table.check_convergence():
        v_table.render()
        print("R_avg =", np.sum(v_table.v)/(env.H*env.W - 1))
        print('Update times =', v_table.update_times)
        break

gamma = 1
env = GridEnv()
v_table = StateValue(env.H, env.W)
max_iteration = 2000
episode_num = 1000
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
    if v_table.check_convergence():
        v_table.render()
        print("R_avg =", np.sum(v_table.v)/(env.H*env.W - 1))
        print('Update times =', v_table.update_times)
        break




