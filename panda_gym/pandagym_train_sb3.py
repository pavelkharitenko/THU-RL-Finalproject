from stable_baselines3 import DDPG
import gymnasium as gym
from datetime import datetime
import panda_gym
import randomname
import time

# first create unique experiment name with experiment parameters:
human_name = randomname.get_name()
dt_string = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
# experiment_parameters
max_timesteps =  10_000
alg_name = DDPG.__name__
env_name = ["PandaReachDense-v3", "PandaReach-v3", 
            "PandaReachJointsDense-v3", "PandaReachJoints-v3"][3]

experiment_name = dt_string + "-" + env_name + "-" + alg_name + "-" + str(max_timesteps) + "ts" + "-" + human_name

print(experiment_name)

logdir = "pandatrain_logs"
modeldir = "./trained_rl_agents/"
# given by environment:
# max episode length: 50




env = gym.make(env_name, render_mode="human")
model = DDPG(policy="MultiInputPolicy", env=env, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=max_timesteps, tb_log_name=experiment_name)

model.save(modeldir + experiment_name)

# visualize agent
print("##### training finished ######")
time.sleep(2.0)

observation, info = env.reset()

#vec_env = model.get_env()
#obs = vec_env.reset()
for i in range(500):
    action, _state = model.predict(observation, deterministic=True)
    #obs, reward, done, info = vec_env.step(action)
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
    time.sleep(0.01)
    #vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

#print("first rew", str(rewards_list[0][0]))
print("#### End #####")
exit(0)
for _ in range(500):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
    #time.sleep(0.01)

time.sleep(2.0)