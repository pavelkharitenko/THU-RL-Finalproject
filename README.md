## THU SIGS RL Spring 2024, Final Project Group 2 - Robotic Manipulation

### Installation:

Install through pip the packages `stable_baselines3`and `gymnasium`, `imitation` and `pandagym` according to their docs. (Also install `seaborn` package if you want to run the plot script)

Can be installed into a python environment like conda.


### Run Experiment

After installing (and activating conda environment, if any), run a script with `python3 pandagym_test_agent.py`.

### Alternative options

Run in Google Colab (install sb3, gymnasium, panda, etc. and copy code to there) if it is not working on OS besides Ubuntu.


### Available scripts:

- `pandagym_train_sb3.py` - to train RL agent in pandagym environment, can be visualized and saved

- `pandagym_test_agent.py` - to test your saved RL agent in pandagym env.

- `pandagym_imitation_learning.py` - pretrain and save an RL agent on expert demonstrations via BC.

- `pandagym_train_student.py` - continue training a pretrained RL agent on expert demonstrations.

- `pandagym_curriculum_learning.py` - decreasing threshold for Reach task using the sparse-end-effector configuation.

- `plot_panda_gym_results.py` - to plot results after training, if agent was saved

- `pandagym_train_demo.py` - no use, just minimal demo code for presenting training agent in gym

- `gym_classiccontrol_sb3.py` - to just try out gymnasium and sb3 libraries.


Other scripts are just similar and do some of the functions above for multiple environmens/algorithms at once.


### Agents:

Agents plotted and used in the final report are in `./panda_gym/pg_agents/final_agents`.

Their respective tensorboard logs are in `./panda_gym/pg_logs` with the same last name.



