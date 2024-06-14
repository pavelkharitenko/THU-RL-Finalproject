from stable_baselines3 import DDPG
import gymnasium as gym
import panda_gym
import time
import numpy as np
#from panda_curriculum_learning import CurriculumPandaReachEnv

env = gym.make('PandaReachDense-v3', render_mode="human")
#agent_path = "panda_gym/pg_agents/final_agents/2024.06.12-22:38:32-PandaReachDense-v3-DDPG-10000ts-champagne-sandcrab"

agent_path = "panda_gym/pg_agents/midterm_agents/2024.05.14-20:26:13-PandaReach-v3-DDPG-10000ts-muted-radio"


model = DDPG.load(agent_path, policy="MultiInputPolicy", env=env, verbose=1)


observation, info = env.reset()

print("info:", info)

time.sleep(2.0)
for _ in range(500):
    #action = env.action_space.sample() # random action
    action, _state = model.predict(observation, deterministic=True)

    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print("info:", info)
        print("reward for iteration",_, reward)
        observation, info = env.reset()
        time.sleep(2)
    time.sleep(0.03)

time.sleep(2.0)