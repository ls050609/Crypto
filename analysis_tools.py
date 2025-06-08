import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats as stats

# 1. 频率分布测试
def frequency_test(data, N):
    counts = np.bincount(data, minlength=N)
    probs = counts / len(data)
    plt.bar(range(N), probs)
    plt.axhline(y=1/N, color='r', linestyle='--', label='Expected')
    plt.title(f"Frequency Distribution (N={N})")
    plt.xlabel("Number")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig('frequency_plot.png')
    plt.close()
    return probs

# 2. 间隔分布测试
def interval_test(intervals_dict):
    plt.figure(figsize=(12, 6))
    for num, intervals in intervals_dict.items():
        if intervals:
            plt.hist(intervals, bins=50, alpha=0.5, label=f'Number {num}')
    plt.title("Interval Distribution")
    plt.xlabel("Interval")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig('interval_distribution.png')
    plt.close()

# 3. 卡方检验
def chi_square_test(data, N):
    observed = np.bincount(data, minlength=N)
    expected = np.full(N, len(data)/N)
    chi2, p = stats.chisquare(observed, expected)
    return chi2, p

# 4. 自相关测试
def autocorrelation_test(data, max_lag=20):
    acf = []
    for lag in range(1, max_lag+1):
        corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
        acf.append(corr)
    
    plt.bar(range(1, max_lag+1), acf)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Autocorrelation Test")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.savefig('autocorrelation.png')
    plt.close()
    return acf

# 5. 游程测试
def runs_test(data):
    median = np.median(data)
    binary_data = [1 if x >= median else 0 for x in data]
    
    runs = []
    current_run = 1
    for i in range(1, len(binary_data)):
        if binary_data[i] == binary_data[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    
    plt.hist(runs, bins=20)
    plt.title("Runs Test")
    plt.xlabel("Run Length")
    plt.ylabel("Frequency")
    plt.savefig('runs_test.png')
    plt.close()
    
    # 游程数统计检验
    n = len(binary_data)
    n1 = sum(binary_data)
    n0 = n - n1
    runs_count = len(runs)
    
    expected_runs = (2 * n0 * n1) / n + 1
    var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
    z = (runs_count - expected_runs) / np.sqrt(var_runs)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return runs, z, p

# 主测试函数
def main_analysis(N=10):
   
    data = np.loadtxt('random_numbers.txt', dtype=int)
    
    # 读取间隔数据
    intervals_dict = {}
    with open('intervals.txt', 'r') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            num = int(parts[0])
            if parts[1].strip():
                intervals = list(map(int, parts[1].split()))
                intervals_dict[num] = intervals
    
    print("1. 频率测试...")
    probs = frequency_test(data, N)
    print("概率分布:", probs)
    
    print("\n2. 间隔分布测试...")
    interval_test(intervals_dict)
    
    print("\n3. 卡方检验...")
    chi2, p = chi_square_test(data, N)
    print(f"卡方值: {chi2:.4f}, p值: {p:.4f}")
    print(f"显著性: {'不显著 (均匀分布)' if p > 0.05 else '显著 (非均匀分布)'}")
    
    print("\n4. 自相关测试...")
    acf = autocorrelation_test(data)
    print(f"自相关系数 (前5个): {acf[:5]}")
    
    print("\n5. 游程测试...")
    runs, z, p_val = runs_test(data)
    print(f"游程数: {len(runs)}, Z值: {z:.4f}, p值: {p_val:.4f}")
    print(f"随机性: {'符合' if p_val > 0.05 else '不符合'}")

if __name__ == "__main__":
    main_analysis()