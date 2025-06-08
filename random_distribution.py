import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def test_random_distribution(N, sample_size):
    # 初始化数据结构
    frequency = [0] * N
    positions = defaultdict(list)
    intervals = defaultdict(list)
    random_sequence = []
    
    # 生成随机数序列并记录位置
    for i in range(sample_size):
        num = random.randint(0, N-1)
        random_sequence.append(num)
        frequency[num] += 1
        positions[num].append(i)
    
    # 计算间隔分布
    for num in range(N):
        if len(positions[num]) > 1:
            for j in range(1, len(positions[num])):
                interval = positions[num][j] - positions[num][j-1]
                intervals[num].append(interval)
    
    # 保存结果到文件
    with open('frequency.txt', 'w') as freq_file:
        freq_file.write("Number\tFrequency\tProbability\n")
        for i in range(N):
            prob = frequency[i] / sample_size
            freq_file.write(f"{i}    {frequency[i]}    {prob:.6f}\n")
    
    with open('intervals.txt', 'w') as interval_file:
        interval_file.write("Number\tIntervals\n")
        for i in range(N):
            interval_file.write(f"{i}\t")
            if i in intervals:
                interval_file.write(" ".join(map(str, intervals[i])))
            interval_file.write("\n")
    
    with open('random_numbers.txt', 'w') as num_file:
        for num in random_sequence:
            num_file.write(f"{num}\n")
    
    # 可视化频率分布
    plt.figure(figsize=(10, 6))
    plt.bar(range(N), frequency)
    plt.axhline(y=sample_size/N, color='r', linestyle='--', label='Expected')
    plt.title(f"Frequency Distribution (N={N}, Sample Size={sample_size})")
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig('frequency_distribution.png')
    plt.close()
    
    print(f"测试完成，结果已保存到文件。样本大小: {sample_size}")

# 参数设置
N = 10            # 随机数范围 [0, N-1]
sample_size = 100000  # 样本大小

# 执行测试
test_random_distribution(N, sample_size)