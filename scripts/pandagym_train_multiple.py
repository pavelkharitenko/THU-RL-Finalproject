from stable_baselines3 import PPO, DDPG, TD3, SAC
import gymnasium as gym
from datetime import datetime
import panda_gym, time, randomname



# Can be run just for training, and visualizing, or run with saving model and logging.

SaveModelandLogs = True # Set to True if agent model needs to be saved after training.

# select experiment_parameters
max_timesteps =  30_000

# choose pandagym environment (reach/pickandplace/etc., dense/sparse, joint/endeffector)

algs = [TD3]


# ee-control and dense/sparse
selected_environments = ["PandaReach-v3"]


for rl_alg in algs:
    alg_name = rl_alg.__name__
    for rl_env in selected_environments:
        env_name = rl_env




        # create experiment name like "2024.05.14-20:26:13-PandaReach-v3-DDPG-10000ts-muted-radio_1"
        experiment_name = datetime.now().strftime("%Y.%m.%d-%H:%M:%S") + "-" + env_name + "-" \
                        + alg_name + "-" + str(max_timesteps) + "ts" + "-" + randomname.get_name()


        print("Begin experiment ", experiment_name)
        print("Agent and logs will be saved after training:", SaveModelandLogs)

        logdir = "panda_gym/pg_logs" if SaveModelandLogs else None # tensorboard will not store logs if set to none
        modeldir = "./panda_gym/pg_agents/"


        # run training experiment
        env = gym.make(env_name)
        env.reset()
        model = rl_alg(policy="MultiInputPolicy", env=env, verbose=1, tensorboard_log=logdir)
        model.learn(total_timesteps=max_timesteps, tb_log_name=experiment_name)


        # save agent if neededx
        if SaveModelandLogs:
            model.save(modeldir + experiment_name)


exit(0)
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

print("#### Visualization ended #####")

