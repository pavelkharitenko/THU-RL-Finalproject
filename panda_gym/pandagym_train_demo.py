from stable_baselines3 import DDPG
import gymnasium as gym
import panda_gym, time

# train agent
env = gym.make("PandaReachDense-v3", render_mode="human")
model = DDPG(policy="MultiInputPolicy", env=env, verbose=1)
model.learn(total_timesteps=10_000)

model.save("our_DDPG_agent")

print("training ended")

