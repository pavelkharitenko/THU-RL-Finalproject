from stable_baselines3.common.evaluation import evaluate_policy

def create_experiment_name():
    pass


def evaluate_agent(agent_path, alg, eval_env, n_episodes=100):
    model = alg.load(agent_path, policy="MultiInputPolicy", env=eval_env, verbose=1)
    reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=n_episodes)
    print("Agent",agent_path,"evaluated on", n_episodes, " episodes, has reward: ", reward)
