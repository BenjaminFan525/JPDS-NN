import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from utils import COLORS

log_paths = [
        "/home/fanyx/mdvrp/result/training_rl/ppo/2024-12-25__22-52/log.txt",
        "/home/fanyx/mdvrp/result/training_rl/ppo/2024-12-27__12-10/log.txt"
    ]

ori_logs = []

for idx, log_path in enumerate(log_paths):
    log = []
    with open(log_path, 'r') as f:
        for data in f.readlines():
            data = data.strip('\n')
            log.append(list(map(float, data.split(' '))))

    ori_logs.append(log)

ori_logs[1] = ori_logs[1][1:]
filename = '/home/fanyx/mdvrp/experiment/log_c.txt'

with open(filename, 'w') as file:
    for sublist in ori_logs[0]:
        # 将子列表转换为字符串，元素之间用空格分隔，并写入文件
        # 然后写入换行符以开始新的一行
        file.write(' '.join(map(str, sublist)) + '\n')

    for sublist in ori_logs[1]:
        # 将子列表转换为字符串，元素之间用空格分隔，并写入文件
        # 然后写入换行符以开始新的一行
        sublist[1] += ori_logs[0][-1][1]
        sublist[2] += ori_logs[0][-1][2]
        file.write(' '.join(map(str, sublist)) + '\n')