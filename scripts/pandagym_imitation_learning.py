import numpy as np
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import time, randomname
from datetime import datetime
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
import gymnasium as gym
import panda_gym

rng = np.random.default_rng(0)

# Experiment params

SaveModel = True # Set to True if agent model needs to be saved after training.
n_epochs_bc_pretrain = 100
alg_name = PPO.__name__
env_name = ["PandaReach-v3"][0]

experiment_name = datetime.now().strftime("%Y.%m.%d-%H:%M:%S") + "-" + env_name + "-" \
                + alg_name + "-" + str(n_epochs_bc_pretrain) + "ts" + "-" + randomname.get_name()

modeldir = "./panda_gym/pg_pretrained_agents/"

env = make_vec_env(
    "PandaReach-v3",
    #"CartPole-v1",
    rng=rng,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)

def sample_expert_transitions(expert):
    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)


def load_expert():
    print("loading a pretrained expert.")
    expert = load_policy(
        "sac",
        env_name="PandaReachDense-v3",
        venv=env,
    )
    return expert



# Load expert 
expert_agent_path = "panda_gym/pg_agents/report_agents/2024.06.14-19:19:06-PandaReach-v3-DDPG-50000ts-distinct-chart"
expert = DDPG.load(expert_agent_path, policy="MultiInputPolicy", env=env, verbose=1)

# Evaluate expert
#rew_mean, rew_std = evaluate_policy(expert, env, 1000)
#print("##### Expert Reward:", rew_mean, rew_std)

# Create student
naive_agent = PPO(policy="MultiInputPolicy", env=env, verbose=1)

# Evaluate student
#rew_mean, rew_std = evaluate_policy(naive_agent, env, 1000)
#print("##### Naive Agent Reward:", rew_mean, rew_std)


# Collect trajectories from expert
transitions = sample_expert_transitions(expert)
print("transitions: ", len(transitions))


# Setup BC trainer
bc_trainer = bc.BC(
    policy=naive_agent.policy,
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)

evaluation_env = make_vec_env(
    "PandaReach-v3",
    rng=rng
)


# Evaluate naive policy
print("Evaluating the untrained policy.")
#rew_mean, rew_std = evaluate_policy(bc_trainer.policy, evaluation_env, n_eval_episodes=1000)
#print("##### Naive Policy Reward:", rew_mean, rew_std)


""" # visualize agent for 500 timesteps
print("##### training finished, begin visualizing agent... #####")
env = gym.make("PandaReachDense-v3", render_mode="human")
observation, info = env.reset()

for i in range(500):
    action, _state = naive_agent.predict(observation, deterministic=True)
    
    # not sure how to plot or utilize these variables yet
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
    time.sleep(0.01)
print("#### Visualization ended #####") """


# Train student with BC

bc_trainer.train(n_epochs=n_epochs_bc_pretrain, log_interval=10)
#reward, _ = evaluate_policy(bc_trainer.policy, evaluation_env, 1000)

# Evaluate student after BC
#print("Naive agent after pretraining Reward:", reward)


# Save if needed
if SaveModel:
    naive_agent.save(modeldir + experiment_name)
    print("saved pretrained agent ", experiment_name)

print("-----------------------------------")
exit(0)





