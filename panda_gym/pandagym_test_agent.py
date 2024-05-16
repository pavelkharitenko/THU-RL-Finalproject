from stable_baselines3 import DDPG
import gymnasium as gym
import panda_gym
import time
import numpy as np


env = gym.make('PandaReachJoints-v3', render_mode="human")
model = DDPG.load("trained_rl_agents/2024.05.14-20:56:59-PandaReachJoints-v3-DDPG-10000ts-odious-redoubt", policy="MultiInputPolicy", env=env, verbose=1)


observation, info = env.reset()

time.sleep(2.0)
for _ in range(500):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(reward)
        observation, info = env.reset()
    time.sleep(0.03)

time.sleep(2.0)