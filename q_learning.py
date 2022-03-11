
from matplotlib.pyplot import axis
from Env import GridEnv
import numpy as np



class ActionValue:
    def __init__(self, H, W, action_num, seed=1) -> None:
        self.q = np.zeros((H, W, action_num))
        self.alpha = 0.5
        self.gamma = 0.9

    def get_action(self, obs):
        return np.random.randint(4)

    def update(self, obs, action, reward, next_obs):
        self.q[obs[0], obs[1], action] = (1-self.alpha)*self.q[obs[0], obs[1], action] + self.alpha * (reward + self.gamma * np.max(self.q[next_obs[0], next_obs[1]]))

    def render(self):
        print(np.argmax(self.q, axis=-1))

max_iteration = 200
episode_num = 100
env = GridEnv()
q = ActionValue(env.H, env.W, env.action_num)
for i in range(max_iteration):
    for i in range(episode_num):
        obs = env.reset()
        action = q.get_action(obs)
        next_obs, reward, done, info = env.step(action)
        q.update(obs, action, reward, next_obs)
        if done:
            break
    q.render()

    