from stable_baselines3 import DDPG, TD3, SAC, PPO
import gymnasium as gym
from datetime import datetime
import panda_gym, time, randomname
from utils import evaluate_agent
from imitation.util.util import make_vec_env
import numpy as np
# Can be run just for training, and visualizing, or run with saving model and logging.
SaveModelandLogs = True # Set to True if agent model needs to be saved after training.

# select experiment_parameters
max_timesteps =  20_000
alg_name = PPO.__name__

# choose pandagym environment (reach/pickandplace/etc., dense/sparse, joint/endeffector)
env_name = ["PandaReachDense-v3", "PandaReach-v3", "PandaReachJointsDense-v3", "PandaReachJoints-v3",
            "PandaPushDense-v3"][1]

pretrained_agent_path = "panda_gym/pg_pretrained_agents/2024.06.21-15:07:08-PandaReach-v3-PPO-100ts-sluggish-foot"
pretrained_agent_name = "2024.06.21-15:07:08-PandaReach-v3-PPO-100ts-sluggish-foot"

# create experiment name like "2024.05.14-20:26:13-PandaReach-v3-DDPG-10000ts-muted-radio_1"
experiment_name = pretrained_agent_name + "-" + "refined-for"+ env_name + "-" + str(max_timesteps) + "ts"


print("Begin experiment ", experiment_name)
print("Agent and logs will be saved after training:", SaveModelandLogs)

logdir = "panda_gym/pg_logs" if SaveModelandLogs else None # tensorboard will not store logs if set to none
modeldir = "./panda_gym/pg_agents/"


# run training experiment
env = gym.make(env_name)
env.reset()
model = PPO.load(pretrained_agent_path, policy="MultiInputPolicy", env=env, verbose=1, tensorboard_log=logdir)
# start train pretrained agent
model.learn(total_timesteps=max_timesteps, tb_log_name=experiment_name)

# save agent if needed
if SaveModelandLogs:
    model.save(modeldir + experiment_name)


# visualize agent for 500 timesteps
print("##### training finished, begin visualizing agent... ######")
time.sleep(2.0)

observation, info = env.reset()

for i in range(500):
    action, _state = model.predict(observation, deterministic=True)
    
    # not sure how to plot or utilize these variables yet
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
    time.sleep(0.01)


rng = np.random.default_rng(0)
evaluation_env = make_vec_env(
    "PandaReach-v3",
    rng=rng
)

evaluate_agent("panda_gym/pg_agents/report_agents/2024.06.16-23:38:46-PandaReach-v3-TD3-35000ts-quiet-store", TD3, evaluation_env, 1000)

print("#### Visualization ended #####")

