import numpy as np
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy


from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

import gymnasium as gym

import panda_gym

rng = np.random.default_rng(0)
env = make_vec_env(
    "PandaReachDense-v3",
    #"CartPole-v1",
    rng=rng,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)



def train_expert():
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training a expert.")
    
    expert = DDPG(policy="MultiInputPolicy", env=env, verbose=1)
    expert.learn(10)  # Note: change this to 100_000 to train a decent expert.
    return expert




def sample_expert_transitions():
    expert = train_expert()  # uncomment to train your own expert
    #expert = download_expert()

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

expert_agent_path = "panda_gym/pg_agents/2024.06.14-23:20:48-PandaReachDense-v3-DDPG-16500ts-formal-hertz"


model = DDPG.load(expert_agent_path, policy="MultiInputPolicy", env=env, verbose=1)
#expert = load_expert()  # uncomment to train your own expert
expert = model

reward, _ = evaluate_policy(expert, env, 10)
print("##### Reward:", reward)



exit(0)
transitions = sample_expert_transitions()
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)

print(transitions)

evaluation_env = make_vec_env(
    "CartPole-v1",
    rng=rng,
    env_make_kwargs={"render_mode": "human"},  # for rendering
)




print("done.")
exit(0)
def download_expert():
    print("Downloading a pretrained expert.")
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="CartPole-v1",
        venv=env,
    )
    return expert