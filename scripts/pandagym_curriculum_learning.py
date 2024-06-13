import time, randomname
from datetime import datetime
from panda_gym.envs.panda_tasks import PandaReachEnv
import numpy as np
from stable_baselines3 import DDPG
import gymnasium as gym

# custom reach task which inits goal
class CurriculumPandaReachEnv(PandaReachEnv):
    """Custom Reach task wih Panda robot, which inits the goal closer to gripper initially.
    """

    def __init__(self, render_mode, control_type, reward_type):
        super().__init__(render_mode=render_mode, control_type=control_type, reward_type=reward_type)
        self.task.reset = self.reset_curricula
        self.distance_factor = 8.0

        self.goal_reached_counter = 0
        self.total_goal_reached_counter = 0

        #local_goal_range = 0.1
        #self.local_goal_range_low = np.array([-local_goal_range / 2, -local_goal_range / 2, 0])
        #self.local_goal_range_high = np.array([local_goal_range / 2, local_goal_range / 2, local_goal_range])
        
        
        self.task.distance_threshold = 0.18
        self.task.sim.physics_client.removeBody(self.task.sim._bodies_idx["target"])
        self.sim.create_sphere(
            body_name="target",
            radius=self.task.distance_threshold,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def reset_curricula(self):
        #print(self.task.get_ee_position())
        #sampled_goal = self.sample_local_goal_position()
        #self.task.goal = self.task.get_ee_position() + sampled_goal + np.array([0.0, 0.0, -.2])

        self.task.goal = self.task._sample_goal() * self.distance_factor*self.task.distance_threshold
        self.task.sim.set_base_pose("target", self.task.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        

        if self.goal_reached_counter == 8:
            self.update_target()
            self.goal_reached_counter = 0
        

    def update_target(self):
        if self.task.distance_threshold < 0.02:
            self.task.distance_threshold = 0.02
            self.distance_factor = 50.0
        else: 
            self.task.distance_threshold -= 0.03
        print("updating target position to", self.task.distance_threshold)
        self.task.sim.physics_client.removeBody(self.task.sim._bodies_idx["target"])
        self.sim.create_sphere(
            body_name="target",
            radius=self.task.distance_threshold,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def reset(self, seed = None, options = None):
        
        obs, info = super().reset(seed=seed, options=options)
        if info["is_success"]:
            self.goal_reached_counter += 1
            print("totally reached since last increase:", self.goal_reached_counter)
            self.total_goal_reached_counter += 1
            print("total reaches in whole trainig:", self.total_goal_reached_counter)

        return obs, info
    


    def sample_local_goal_position(self):

        goal = self.task.np_random.uniform(self.local_goal_range_low, self.local_goal_range_high)
        return goal


# register and test custom gym env 
from gymnasium.envs.registration import register

register(
    id="CurriculumPandaReachEnv-v3",
    entry_point=CurriculumPandaReachEnv,
    kwargs={"reward_type": "dense", "control_type": "joints"},
    max_episode_steps=50,
)


SaveModelandLogs = False # Set to True if agent model needs to be saved after training.

# select experiment_parameters
max_timesteps = 20_000
alg_name = DDPG.__name__
# choose pandagym environment (reach/pickandplace/etc., dense/sparse, joint/endeffector)
env_name = "CurriculumPandaReachEnv-v3"

# create experiment name like "2024.05.14-20:26:13-PandaReach-v3-DDPG-10000ts-muted-radio_1"
experiment_name = datetime.now().strftime("%Y.%m.%d-%H:%M:%S") + "-" + env_name + "-" \
                + alg_name + "-" + str(max_timesteps) + "ts" + "-" + randomname.get_name()


print("Begin experiment ", experiment_name)
print("Agent and logs will be saved after training:", SaveModelandLogs)

logdir = "panda_gym/pandatrain_logs" if SaveModelandLogs else None # tensorboard will not store logs if set to none
modeldir = "./panda_gym/trained_rl_agents/"



# run training experiment
gym_env = gym.make("CurriculumPandaReachEnv-v3", render_mode="human")
model = DDPG(policy="MultiInputPolicy", env=gym_env, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=max_timesteps, tb_log_name=experiment_name)





# save agent if needed
if SaveModelandLogs:
    model.save(modeldir + experiment_name)




exit(0)
observation, info = gym_env.reset()

for _ in range(500):
    action = gym_env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = gym_env.step(action)

    if terminated or truncated:
        observation, info = gym_env.reset()
        print(" ####### custom gym env got reset #######")
        time.sleep(4)
    
    time.sleep(0.03)

print("finished testing custom env")