
from Env import GridEnv
import numpy as np



class ActionValue:
    def __init__(self, H, W, action_num, seed=1) -> None:
        self.action_num = action_num
        self.q = np.zeros((H, W, self.action_num))
        self.tem_q = self.q.copy()
        self.alpha = 0.5
        self.gamma = 0.9

    def get_action(self, obs):
        return np.random.randint(4)

    def update(self, obs, action, reward, next_obs):
        self.q[obs[0], obs[1], action] = (1-self.alpha)*self.q[obs[0], obs[1], action] + self.alpha * (reward + self.gamma * np.max(self.q[next_obs[0], next_obs[1]]))

    def render(self):
        print(np.argmax(self.q, axis=-1))
        print(self.q.reshape(-1, self.action_num))
        print(self.tem_q.reshape(-1, self.action_num))

    def check_convergence(self):
        done = np.allclose(self.q, self.tem_q, atol=0.0001)
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

    