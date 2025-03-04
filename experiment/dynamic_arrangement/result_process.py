import os
import numpy as np

# 指定目录路径
directory = 'experiment/dynamic_arrangement/2025-02-22__20-29_field_ga'

result = [[] for _ in range(6)]

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # 检查文件扩展名是否为.txt
    if filename.endswith('.txt'):
        # 构建完整的文件路径
        file_path = os.path.join(directory, filename)
        # 打开并读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                # 去除行尾的换行符并分割行内容
                elements = line.strip().split()
                # 将分割后的字符串转换为所需的数据类型，例如整数
                result[idx] += [float(element) for element in elements if element != 'nan']

print(f"[mdvrp]  | Distance:{np.mean(result[0]):.2f} | Time: {np.mean(result[1]):.2f} | Fuel: {np.mean(result[2]):.2f}")
print(f"[GA]     | Distance:{np.mean(result[3]):.2f} | Time: {np.mean(result[4]):.2f} | Fuel: {np.mean(result[5]):.2f}")