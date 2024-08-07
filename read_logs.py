#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt

id = "2024-08-06_16-34"
with open(f'logs_obs_{id}.json', 'r') as f:
    obs = json.load(f)

with open(f'logs_action_{id}.json', 'r') as f:
    actions = json.load(f)

# Create a figure with multiple subplots
fig, axs = plt.subplots(len(obs), figsize=(10, 20))

# Iterate over each joint in the log
for i, (joint, data) in enumerate(obs.items()):
    time = range(len(data))
    axs[i].plot(time, data,label='Observation')
    axs[i].plot(time, actions[joint], label='Action')
    axs[i].set_xlabel('Steps')
    axs[i].set_ylabel(joint)
    axs[i].legend() 
    axs[i].grid(True)

plt.tight_layout()
plt.savefig(f'jt_logs_{id}.png')


# ########### action ##############



# # Create a figure with multiple subplots
# fig, axs = plt.subplots(len(existing_log), figsize=(10, 20))

# # Iterate over each joint in the log
# for i, (joint, data) in enumerate(existing_log.items()):
#     time = range(len(data))
#     axs[i].plot(time, data)
#     axs[i].set_xlabel('Steps')
#     axs[i].set_ylabel(joint)
#     axs[i].grid(True)

# plt.tight_layout()
# plt.savefig(f'jt_action_{id}.png')


