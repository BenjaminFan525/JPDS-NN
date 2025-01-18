import os
import numpy as np

# 指定目录路径
directory = '/home/fanyx/mdvrp/experiment/dynamic_arrangement/2025-01-18__21-21'

result_mdvrp = [[] for _ in range(6)]
result_ia = [[] for _ in range(6)]

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
                if 'mdvrp' in filename:
                    result_mdvrp[idx] += [float(element) for element in elements]
                else:
                    result_ia[idx] += [float(element) for element in elements]


print(f"[mdvrp-ori] | Distance:{np.mean(result_mdvrp[0]):.2f} | Time: {np.mean(result_mdvrp[1]):.2f} | Fuel: {np.mean(result_mdvrp[2]):.2f}")
print(f"[mdvrp-rec] | Distance:{np.mean(result_mdvrp[3]):.2f} | Time: {np.mean(result_mdvrp[4]):.2f} | Fuel: {np.mean(result_mdvrp[5]):.2f}")

print(f"[ia-ori]    | Distance:{np.mean(result_ia[0]):.2f} | Time: {np.mean(result_ia[1]):.2f} | Fuel: {np.mean(result_ia[2]):.2f}")
print(f"[ia-rec]    | Distance:{np.mean(result_ia[3]):.2f} | Time: {np.mean(result_ia[4]):.2f} | Fuel: {np.mean(result_ia[5]):.2f}")