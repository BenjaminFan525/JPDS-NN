import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 创建示例数据
data = {
    'N_fields': [2, 3, 4, 5, 6],
    'JPDS-NN': [2483.05, 3210.31, 3953.39, 4666.80, 5328.86],
    'GA': [2353.77, 3243.28, 4193.80, 4960.29, 6140.87],
}

df = pd.DataFrame(data)

# 设置全局科学计数法
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.formatter.limits'] = (-4, 3) 
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# sns.set_theme(style="whitegrid", font_scale=2.5)

# 创建图形和轴
plt.figure(figsize=(8, 8))

# sns.lineplot(x='N_fields', y='RA', data=df, marker='o', label='RA', color='grey', linewidth=2)
plt.plot(data['N_fields'], data['GA'], marker='o', label='OGA-t', color='orange', markersize=15,linewidth=4)
plt.plot(data['N_fields'], data['JPDS-NN'], marker='s', label='JPDS-nn-t', color='navy', markersize=15, linewidth=4)

# 添加图例
plt.legend(fontsize=18)
plt.grid()

# 添加标题和标签
# plt.title('Test Results of Fuel Consumption')
plt.xlabel('Number of Plots', fontsize=18)
plt.ylabel('Time (s)', fontsize=18)
plt.xticks([2, 3, 4, 5, 6], ['2', '3', '4', '5', '6'])

# 显示图形
save_dir = '/home/fanyx/mdvrp/experiment/test_t.png'
plt.savefig(save_dir)
print(save_dir)