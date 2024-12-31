import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from utils import COLORS

log_paths = [
        "/home/fanyx/mdvrp/experiment/log_s.txt",
        "/home/fanyx/mdvrp/experiment/log_c.txt",
        "/home/fanyx/mdvrp/experiment/log_t.txt"
    ]

labels = [
    "PPO-s",
    "PPO-c",
    "PPO-t",
]

plt.figure(0)
plt.figure(1)
plt.figure(2)
for idx, (log_path, label) in enumerate(zip(log_paths, labels)):
    log = []
    with open(log_path, 'r') as f:
        for data in f.readlines():
            data = data.strip('\n')
            log.append(list(map(float, data.split(' '))))

    log = np.array(log)

    plt.figure(0)
    plt.plot(
        log[:, 2],
        log[:, 3],
        alpha=0.3, 
        color=COLORS[idx+3],
        # label = label
    )
    plt.plot(
        log[:, 2],
        signal.savgol_filter(log[:, 3], 31, 3),
        color=COLORS[idx+3],
        linewidth=2,
        label = label
    )

    plt.figure(1)
    plt.plot(
        log[:, 2],
        log[:, 4],
        alpha=0.3, 
        color=COLORS[idx+3],
        # label = label
    )
    plt.plot(
        log[:, 2],
        signal.savgol_filter(log[:, 4], 31, 3),
        color=COLORS[idx+3],
        linewidth=2,
        label = label
    )

    plt.figure(2)
    plt.plot(
        log[:, 2],
        log[:, 5],
        alpha=0.3, 
        color=COLORS[idx+3],
        # label = label
    )
    plt.plot(
        log[:, 2],
        signal.savgol_filter(log[:, 5], 31, 3),
        color=COLORS[idx+3],
        linewidth=2,
        label = label
    )

plt.figure(0)

plt.grid()
# plt.ylim((3500, 33000))
plt.xlabel('Number of Episodes', fontsize=18)
plt.ylabel('Distance (m)', fontsize=18)
plt.title('Average Distance', fontsize=18)

plt.legend(fontsize=18)

output_path = os.path.join(*log_path.split(os.path.sep)[:-1], 'distance.png')
if log_path[0] == os.path.sep:
    output_path = os.path.sep + output_path
plt.savefig(output_path)

plt.figure(1)

plt.grid()
# plt.ylim((2000, 17000))
plt.xlabel('Number of Episodes', fontsize=18)
plt.ylabel('Time (s)', fontsize=18)
plt.title('Average Time', fontsize=18)

plt.legend(fontsize=18)


output_path = os.path.join(*log_path.split(os.path.sep)[:-1], 'time.png')
if log_path[0] == os.path.sep:
    output_path = os.path.sep + output_path
plt.savefig(output_path)

plt.figure(2)

plt.grid()
# plt.ylim((40, 115))
plt.xlabel('Episodes', fontsize=18)
plt.ylabel('Fuel Consumption (L)', fontsize=18)
plt.title('Average Fuel Consumption', fontsize=18)

plt.legend(fontsize=18)


output_path = os.path.join(*log_path.split(os.path.sep)[:-1], 'oil.png')
if log_path[0] == os.path.sep:
    output_path = os.path.sep + output_path
plt.savefig(output_path)

print(output_path)