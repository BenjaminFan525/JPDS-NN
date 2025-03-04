import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
import seaborn as sns

COLORS = (
    [
        '#ff4000',
        '#2aaf87',
        '#0064b2',
    ]
)

log_paths = [
        "/home/fanyx/mdvrp/experiment/Training_result/log_s.txt",
        "/home/fanyx/mdvrp/experiment/log_t_16874536.txt",
        "/home/fanyx/mdvrp/experiment/Training_result/log_c.txt",
    ]

labels = [
    "JPDS-NN-s",
    "JPDS-NN-t",
    "JPDS-NN-c",
]

# 设置全局科学计数法
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.formatter.limits'] = (-4, 4) 
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

def smooth_reward(rewards, window=5):
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    return smoothed

plt.figure(0)
plt.figure(1)
plt.figure(2)
for idx, (log_path, label) in enumerate(zip(log_paths, labels)):
    log = []
    with open(log_path, 'r') as f:
        for data in f.readlines():
            data = data.strip('\n')
            log.append(list(map(float, data.split(' '))))

    log = np.array(log)[:240]

    plt.figure(0)
    plt.plot(
        smooth_reward(log[:, 2]),
        smooth_reward(log[:, 3]),
        # log[:, 3],
        alpha=1, 
        color=COLORS[idx],
        label = label
    )
    # plt.plot(
    #     log[:, 2],
    #     signal.savgol_filter(log[:, 3], 31, 3),
    #     color=COLORS[idx],
    #     linewidth=2,
    #     label = label
    # )

    plt.figure(1)
    plt.plot(
        smooth_reward(log[:, 2]),
        smooth_reward(log[:, 4]),
        # log[:, 4],
        alpha=1, 
        color=COLORS[idx],
        label = label
    )
    # plt.plot(
    #     log[:, 2],
    #     signal.savgol_filter(log[:, 4], 31, 3),
    #     color=COLORS[idx],
    #     linewidth=2,
    #     label = label
    # )

    plt.figure(2)
    plt.plot(
        smooth_reward(log[:, 2]),
        smooth_reward(log[:, 5]),
        # log[:, 5],
        alpha=1, 
        color=COLORS[idx],
        label = label
    )
    # plt.plot(
    #     log[:, 2],
    #     signal.savgol_filter(log[:, 5], 31, 3),
    #     color=COLORS[idx],
    #     linewidth=2,
    #     label = label
    # )

plt.figure(0)

plt.grid()
# plt.ylim((3500, 33000))
plt.xlabel('Timesteps', fontsize=18)
plt.ylabel('Distance (m)', fontsize=18)

plt.legend(fontsize=18)

output_path = os.path.join(*log_path.split(os.path.sep)[:-1], 'diatance.png')
if log_path[0] == os.path.sep:
    output_path = os.path.sep + output_path
plt.savefig(output_path)

plt.figure(1)
plt.grid()

# plt.ylim((2000, 17000))
plt.xlabel('Timesteps', fontsize=18)
plt.ylabel('Time (s)', fontsize=18)
plt.legend(fontsize=18)


output_path = os.path.join(*log_path.split(os.path.sep)[:-1], 'time.png')
if log_path[0] == os.path.sep:
    output_path = os.path.sep + output_path
plt.savefig(output_path)

plt.figure(2)
plt.grid()

# plt.ylim((40, 115))
plt.xlabel('Timesteps', fontsize=18)
plt.ylabel('Fuel Consumption (L)', fontsize=18)

plt.legend(fontsize=18)


output_path = os.path.join(*log_path.split(os.path.sep)[:-1], 'oil.png')
if log_path[0] == os.path.sep:
    output_path = os.path.sep + output_path
plt.savefig(output_path)

print(output_path)