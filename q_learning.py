
from gym import Env
from Env import GridEnv
import numpy as np



class ActionValue:
    def __init__(self, H, W, action_num, seed=0) -> None:
        self.update_times = 0
        self.rng = np.random.default_rng(seed=seed)
        self.action_num = action_num
        self.q = np.zeros((H, W, self.action_num))
        self.tem_q = self.q.copy()
        self.alpha = 0.5
        self.gamma = 0.9

    def get_action(self, obs):
        return self.rng.integers(4)

    def get_action_with_policy(self, obs):
        return np.argmax(self.q[obs[0], obs[1]])

    def update(self, obs, action, reward, next_obs):
        self.update_times += 1
        self.q[obs[0], obs[1], action] = (1-self.alpha)*self.q[obs[0], obs[1], action] + self.alpha * (reward + self.gamma * np.max(self.q[next_obs[0], next_obs[1]]))

    def render(self):
        print(np.argmax(self.q, axis=-1))
        print(self.q.reshape(-1, self.action_num))
        # print(self.tem_q.reshape(-1, self.action_num))
        print('Update times = ', self.update_times)

    def check_convergence(self):
        done = np.allclose(self.q, self.tem_q, atol=0.00001)
        self.tem_q = self.q.copy()
        return done


max_iteration = 20000
episode_num = 1000
env = GridEnv()
q_table = ActionValue(env.H, env.W, env.action_num)
for i in range(max_iteration):
    for j in range(episode_num):
        obs = env.reset()
        action = q_table.get_action(obs)
        next_obs, reward, done, info = env.step(action)
        q_table.update(obs, action, reward, next_obs)
        if done:
            break
    if q_table.check_convergence():
        # print("R_avg=", np.sum(q_table.q)/(env.H*env.W - 1))
        q_table.render()
        break

return_mat_total = np.zeros((4,4))
env = GridEnv()
eva_num = 1000
for e in range(eva_num):
    return_mat = np.zeros((4,4))
    for i in range(env.H):
        for j in range(env.W):
            if i==env.H-1 and j==env.W-1:
                pass
            else:
                obs = np.array([i,j])
                env.set_obs(obs)
                while True:
                    action = q_table.get_action_with_policy(obs)
                    next_obs, reward, done, info = env.step(action)
                    obs = next_obs
                    return_mat[i,j] += reward
                    if done:
                        break 
    return_mat_total += return_mat
print(return_mat_total/eva_num)
print('Average return is :', sum(return_mat_total.flatten())/eva_num/(env.H*env.W - 1))


        


    