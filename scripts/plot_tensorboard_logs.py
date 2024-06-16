#import tensorflow as tf
import matplotlib.pyplot as plt
from stable_baselines3 import results_plotter


# color map:
# ppo: #7cb342
# sac: #9334E6
# ddpg: #E52592
# td3: #F3BB13

exit(0)
import numpy as np

def get_scalar_run_tensorboard(tag, filepath):
    values,steps = [],[]
    for e in tf.compat.v1.train.summary_iterator(filepath):
        if len(e.summary.value)>0: #Skip first empty element
            print("tags: ", e.summary.value[0].tag)
            if e.summary.value[0].tag==tag:
                tensor = (e.summary.value[0].tensor)
                value, step = (tf.io.decode_raw(tensor.tensor_content, tf.float32)[0].numpy(), e.step)
                values.append(value)
                steps.append(step)
    return values,steps



val, st = get_scalar_run_tensorboard("time/fps",
                               "pandatrain_logs/2024.06.06-21:50:34-PandaReachJoints-v3-DDPG-555ts-sensitive-halibut_1/events.out.tfevents.1717681836.asus-ZenBook.51658.0")

plt.plot(st, val)
plt.show()