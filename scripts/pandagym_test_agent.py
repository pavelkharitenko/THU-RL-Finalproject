from stable_baselines3 import DDPG, PPO
import gymnasium as gym
import panda_gym
import time
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.util.util import make_vec_env
#from panda_curriculum_learning import CurriculumPandaReachEnv
from utils import evaluate_agent

evaluate_visually = False

env = gym.make('PandaReachDense-v3')
#agent_path = "panda_gym/pg_agents/final_agents/2024.06.12-22:38:32-PandaReachDense-v3-DDPG-10000ts-champagne-sandcrab"

agent_path = "panda_gym/pg_pretrained_agents/2024.06.15-15:33:45-PandaReachDense-v3-PPO-100ts-partial-camera"


model = PPO.load(agent_path, policy="MultiInputPolicy", env=env, verbose=1)


if evaluate_visually:
    observation, info = env.reset()
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




rng = np.random.default_rng(0)
evaluation_env = make_vec_env(
    "PandaReachDense-v3",
    rng=rng
)


evaluate_agent("panda_gym/pg_pretrained_agents/2024.06.15-15:33:45-PandaReachDense-v3-PPO-100ts-partial-camera",
               PPO, evaluation_env, 200)

evaluate_agent("panda_gym/pg_pretrained_agents/2024.06.15-15:33:45-PandaReachDense-v3-PPO-100ts-partial-camera-copy",
               PPO, evaluation_env, 200)

