import os
import numpy as np

# 指定目录路径
directory = '/home/fanyx/mdvrp/experiment/Test_compare/GA/2025-02-06__20-32'

result_mdvrp = [[] for _ in range(3)]
result_ia = [[] for _ in range(3)]

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
                if idx < 3:
                    result_mdvrp[idx] += [float(element) for element in elements]
                else:
                    result_ia[idx - 3] += [float(element) for element in elements]


print(f"[mdvrp] | Distance:{np.mean(result_mdvrp[0]):.2f} | Time: {np.mean(result_mdvrp[1]):.2f} | Fuel: {np.mean(result_mdvrp[2]):.2f}")

print(f"[GA]    | Distance:{np.mean(result_ia[0]):.2f} | Time: {np.mean(result_ia[1]):.2f} | Fuel: {np.mean(result_ia[2]):.2f}")