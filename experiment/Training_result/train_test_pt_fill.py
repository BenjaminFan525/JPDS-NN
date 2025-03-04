import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
import seaborn as sns
import pandas as pd

COLORS = (
    [
        '#ff4000',
        '#2aaf87',
        '#0064b2',
    ]
)

log_paths = [
    [
        "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_t_16874536_pt.txt",
        "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_t_27688426_pt.txt",
        # "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_s_16874536_pt.txt",
        # "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_s_27688426_pt.txt",
        # "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_s_985626713_pt.txt",
        "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_t_6564649_pt.txt",
        "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_t_3407_pt.txt"
    ],
    [
        "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_t_16874536_nopt.txt",
        "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_t_27688426_nopt.txt",
        # "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_s_16874536_nopt.txt",
        # "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_s_27688426_nopt.txt",
        # "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_s_985626713_nopt.txt",
        "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_t_6564649_pt.txt",
        "/home/fanyx/mdvrp/experiment/Training_result/logs_pt/log_t_3407_nopt.txt"
    ],
    ]

labels = [
    "pre-trained",
    "not pre-trained",
]

# 设置全局科学计数法
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.formatter.limits'] = (-4, 4) 
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

plt.figure(0)
plt.figure(1)
plt.figure(2)

# 显示图表
plt.show()

def smooth_reward(rewards, window=3):
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    return smoothed

df = []
for label, log_path in zip(labels, log_paths):
    for path in log_path:
        log = []
        with open(path, 'r') as f:
            for data in f.readlines():
                data = data.strip('\n')
                log.append(list(map(float, data.split(' '))))

        log = np.array(log)[:60]
        
        # df.append(pd.DataFrame({
        #     "step": smooth_reward(log[:, 2]),
        #     "s": smooth_reward(log[:, 3]),
        #     "t": smooth_reward(log[:, 4]),
        #     "c": smooth_reward(log[:, 5]),
        #     "Group": np.array([label for _ in range(60-2)])
        # }))

        df.append(pd.DataFrame({
            "step": log[:, 2],
            "distance": log[:, 3],
            "time": log[:, 4],
            "oil": log[:, 5],
            "Group": np.array([label for _ in range(len(log[:, 2]))])
        }))

        log = np.array(log)[:70]

# 转换为DataFrame
df = pd.concat(df, ignore_index=True)

for obj in ['distance', 'time', 'oil']:
    group_stats = df.groupby(["Group", "step"])[obj].agg(["mean", "std", "count"])
    group_stats = group_stats.reset_index()
    smoothed_means = []
    for group in labels:
        # 提取当前组的原始均值序列
        mask = group_stats["Group"] == group
        group_data = group_stats[mask].sort_values("step")
        mean_values = group_data["mean"].values
        
        # 使用Savitzky-Golay滤波器平滑（窗口长度=5，多项式阶数=2）
        window_length = 5  # 需为奇数，且小于时间点数
        polyorder = 2
        smoothed_mean = savgol_filter(mean_values, window_length, polyorder)
        
        # 保存平滑后的结果
        group_data = group_data.copy()
        group_data["smoothed_mean"] = smoothed_mean
        smoothed_means.append(group_data)

    smoothed_df = pd.concat(smoothed_means)

    smoothed_df["ci_lower"] = smoothed_df["mean"] - 1.96 * (smoothed_df["std"] / np.sqrt(smoothed_df["count"]))
    smoothed_df["ci_upper"] = smoothed_df["mean"] + 1.96 * (smoothed_df["std"] / np.sqrt(smoothed_df["count"]))

    plt.figure(figsize=(7, 7))
    palette = COLORS  # 颜色方案

    for i, group in enumerate(labels):
        group_data = smoothed_df[smoothed_df["Group"] == group]
        time = group_data["step"]
        
        # 绘制平滑后的均值曲线
        plt.plot(
            time,
            group_data["smoothed_mean"],
            color=palette[i],
            linewidth=3,
            label=f"{group}"
        )
        
        # 绘制原始数据的置信区间填充
        plt.fill_between(
            time,
            group_data["ci_lower"],
            group_data["ci_upper"],
            color=palette[i],
            alpha=0.2,
        )
    plt.grid()
    plt.xlabel('Timesteps', fontsize=18)
    if obj == "distance":
        plt.ylabel('Distance (m)', fontsize=18)
    else:
        plt.ylabel('Time (s)', fontsize=18)
    plt.legend(fontsize=18)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{obj}_pt_fill.png')
    plt.savefig(output_path)
    plt.close('all')
    print(output_path)