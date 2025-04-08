import gymnasium as gym
from stable_baselines3 import DDPG
env = gym.make("Hopper-v4",render_mode='human')  
model = DDPG.load("ddpg_hopper_v4")
obs, _ = env.reset()  

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()  

env.close()
