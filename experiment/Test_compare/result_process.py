import os
import numpy as np

# 指定目录路径
directory = '/home/fanyx/mdvrp/experiment/Test_compare/GA/2025-02-10__01-42_GA_large'

result_mdvrp = [[] for _ in range(3)]
result_ga = [[] for _ in range(3)]
result_ra = [[] for _ in range(3)]
result_field = [[] for _ in range(9)]

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
                
                # 将分割后的字符串转换为所需的数据类型，例如整数
                if idx < 3:
                    elements = line.strip().split()
                    result_mdvrp[idx] += [float(element) for element in elements]
                elif idx < 6:
                    elements = line.strip().split()
                    result_ga[idx - 3] += [float(element) for element in elements]
                elif idx < 9:
                    elements = line.strip().split()
                    result_ra[idx - 6] += [float(element) for element in elements]
                elif idx < 15:
                    elements = line.strip().split(',')
                    if len(elements) > 1:
                        result_field[idx - 9].append([float(element) for element in elements])


for idx, result in enumerate(result_field):
    if len(result):
        print(f"Testing result of {idx+1} fields:")
        mean_r = np.mean(np.array(result), axis=0)
        print(f"[RA]    | Distance:{mean_r[6]:.2f} | Time: {mean_r[7]:.2f} | Fuel: {mean_r[8]:.2f}")
        print(f"[mdvrp] | Distance:{mean_r[0]:.2f} | Time: {mean_r[1]:.2f} | Fuel: {mean_r[2]:.2f}")
        print(f"[GA]    | Distance:{mean_r[3]:.2f} | Time: {mean_r[4]:.2f} | Fuel: {mean_r[5]:.2f}")

print(f"Testing result of all fields:")
print(f"[GA]    | Distance:{np.mean(result_ra[0]):.2f} | Time: {np.mean(result_ra[1]):.2f} | Fuel: {np.mean(result_ra[2]):.2f}")
print(f"[mdvrp] | Distance:{np.mean(result_mdvrp[0]):.2f} | Time: {np.mean(result_mdvrp[1]):.2f} | Fuel: {np.mean(result_mdvrp[2]):.2f}")

print(f"[GA]    | Distance:{np.mean(result_ga[0]):.2f} | Time: {np.mean(result_ga[1]):.2f} | Fuel: {np.mean(result_ga[2]):.2f}")