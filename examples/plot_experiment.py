import glob
import os
import pdb

import numpy as np
import matplotlib.pyplot as plt

from smartstart.agents.smartstart import SmartStart
from smartstart.utilities.utilities import Summary, calc_average_reward_training_steps

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Get all summaries in data_directory
fps = glob.glob(data_dir + '/*')  # Get all files in data_dir
summaries_normal = []
summaries_smart_start = []
for fp in fps:
    summary = Summary.load(fp)
    if type(summary.agent) is SmartStart:
        summaries_smart_start.append(summary)
    else:
        summaries_normal.append(summary)

# Plot results
plt.figure()

summaries = [summaries_smart_start, summaries_normal]
labels = ['Smart Start', 'Normal']
colors = ['green', 'blue']
for summary, label, color in zip(summaries, labels, colors):
    iter, average_reward, std = calc_average_reward_training_steps(summary)
    lower = average_reward - std
    upper = average_reward + std
    plt.fill_between(iter, lower, upper, alpha=0.3, color=color)
    plt.plot(iter, average_reward, label=label, color=color)

plt.xlabel('Training steps')
plt.ylabel('Average reward')
plt.legend()
plt.show()