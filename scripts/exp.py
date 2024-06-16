from stable_baselines3 import DDPG, TD3, SAC, PPO
import gymnasium as gym
from datetime import datetime
import panda_gym, time, randomname

env = gym.make("PandaReachDense-v3")
env.reset()
model = DDPG(policy="MultiInputPolicy", env=env, verbose=1)

