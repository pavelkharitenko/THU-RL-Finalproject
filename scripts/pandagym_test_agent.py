from stable_baselines3 import DDPG, PPO, SAC, TD3
import gymnasium as gym
import panda_gym
import time
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.util.util import make_vec_env
#from panda_curriculum_learning import CurriculumPandaReachEnv
from utils import evaluate_agent

evaluate_visually = False
agent_path = "panda_gym/pg_agents/2024.06.21-15:07:08-PandaReach-v3-PPO-100ts-sluggish-foot-refined-forPandaReach-v3-20000ts"


if evaluate_visually:
    env = gym.make('PandaReach-v3', render_mode="human")
    model = PPO.load(agent_path, policy="MultiInputPolicy", env=env, verbose=1)
    observation, info = env.reset()
    time.sleep(10.0)
    ep_return = 0
    for i in range(500):

        action, _state = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        if terminated or truncated:
            print("info:", info)
            print("return for episode =",i, ":", reward)
            observation, info = env.reset()
            ep_return = 0
            
            time.sleep(2)
        time.sleep(0.03)
    time.sleep(2.0)




rng = np.random.default_rng(0)
evaluation_env = make_vec_env(
    "PandaReach-v3",
    rng=rng
)


evaluate_agent("panda_gym/pg_agents/2024.06.21-15:07:08-PandaReach-v3-PPO-100ts-sluggish-foot-refined-forPandaReach-v3-20000ts", PPO, evaluation_env, 1000)

#evaluate_agent("panda_gym/pg_agents/report_agents/2024.06.16-23:38:46-PandaReach-v3-TD3-35000ts-quiet-store", TD3, evaluation_env, 1000)

# distinct chart on sparse rew. reach: -1.702 0.8337841447281182
# formal hertz on sparse rew. reach:  -1.873 0.8847999773960213
# formal hertz on dense rew reach: -0.23457637499999998 0.22027900296340178
# white surface on sparse rew. reach: -47.988 9.732618147240752
# partial camera refined on sparse rew reach: -2.183 1.8454026660867269