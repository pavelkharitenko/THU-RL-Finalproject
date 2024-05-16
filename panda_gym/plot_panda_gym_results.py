from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


result_dir = "pandatrain_logs/"


result_files = []
absolute_results_dir = Path.joinpath(Path.cwd(), result_dir)
p = Path(absolute_results_dir)


for i in p.glob('**/*.csv'):
     result_files.append(str(i.absolute()))




fig, axs = plt.subplots(1, 3, figsize=(22, 5))
#fig.suptitle('Training results for Sparse/Dense and Joint/End-effector Control configurations')

fig.suptitle("  ")

result_files = sorted(result_files)
for result_file in result_files:
    
    df = pd.read_csv(result_file)
    X = df.Step
    Y = df.Value



    if "ep_len_mean" in result_file:
        axs[0].plot(X, Y, label=result_file.split("Panda")[1].split("ts-")[0]) 
        #plt.xticks(rotation = 25) 
        axs[0].set_xlabel('Total timesteps') 
        #plt.ylabel('Values') 
        axs[0].set_title("Episode Length mean") 
        #axs[0].set_title("Posterior of Slope b, mean at " + str(np.around(b_mean, 3)))
        #axs[0].set_xlabel("Slope")
        #axs[0].set_ylabel("Density")


    if "ep_rew_mean" in result_file:
        axs[1].plot(X, Y, label=result_file.split("Panda")[1].split("ts-")[0]) 
        #plt.xticks(rotation = 25) 
        axs[1].set_xlabel('Total timesteps') 
        #plt.ylabel('Values') 
        axs[1].set_title("Episode Reward mean") 

    if "success_rate" in result_file:
        axs[2].plot(X, Y, label=result_file.split("Panda")[1].split("ts-")[0]) 
        #plt.xticks(rotation = 25) 
        axs[2].set_xlabel('Total timesteps') 
        #plt.ylabel('Values') 
        axs[2].set_title("Success rate") 


plt.tight_layout() # Adjust the layout
fig.legend( loc='upper center', 
    #bbox_to_anchor=(0., 1.35),
    ncol=3, )

plt.show() 


exit(0)
# Plot the posterior distribution of slope




