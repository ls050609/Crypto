import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm, chisquare

# 均匀分布随机数生成器（改进版）
def uniform_generator(N, size=10000):
    # 使用拒绝采样确保均匀分布
    max_val = (2**31 - 1) - (2**31 - 1) % N  # 假设RAND_MAX=2^31-1
    result = []
    while len(result) < size:
        r = np.random.randint(0, 2**31-1)
        if r < max_val:
            result.append(r % N)
    return np.array(result)

# 正态分布随机数生成器（Box-Muller变换）
def normal_generator(mean=0, std=1, size=10000):
    u1 = np.random.uniform(size=size)
    u2 = np.random.uniform(size=size)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return z0 * std + mean

# 测试均匀分布
def test_uniform(N=10, size=10000):
    # 标准randint方法
    data_standard = np.random.randint(0, N, size)
    
    # 改进的均匀分布方法
    data_uniform = uniform_generator(N, size)
    
    # 绘制对比图
    plt.figure(figsize=(12, 6))
    
    # 标准方法
    plt.subplot(121)
    counts_std = np.bincount(data_standard, minlength=N)
    probs_std = counts_std / size
    plt.bar(range(N), probs_std)
    plt.axhline(y=1/N, color='r', linestyle='--')
    plt.title(f"Standard randint (N={N})")
    plt.ylim(0, max(probs_std)*1.2)
    
    # 改进方法
    plt.subplot(122)
    counts_uni = np.bincount(data_uniform, minlength=N)
    probs_uni = counts_uni / size
    plt.bar(range(N), probs_uni)
    plt.axhline(y=1/N, color='r', linestyle='--')
    plt.title(f"Improved Uniform (N={N})")
    plt.ylim(0, max(probs_uni)*1.2)
    
    plt.tight_layout()
    plt.savefig('uniform_comparison.png')
    plt.close()
    
    # 卡方检验
    chi2_std, p_std = chisquare(counts_std)
    chi2_uni, p_uni = chisquare(counts_uni)
    
    print(f"标准方法卡方检验: χ² = {chi2_std:.2f}, p = {p_std:.4f}")
    print(f"改进方法卡方检验: χ² = {chi2_uni:.2f}, p = {p_uni:.4f}")

# 测试正态分布
def test_normal(mean=0, std=1, size=10000):
    # 生成正态分布数据
    data = normal_generator(mean, std, size)
    
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.6, label='Generated')
    
    # 绘制理论正态曲线
    x = np.linspace(mean-4*std, mean+4*std, 100)
    plt.plot(x, norm.pdf(x, mean, std), 'r-', lw=2, label='Theoretical')
    
    plt.title(f"Normal Distribution (μ={mean}, σ={std})")
    plt.legend()
    plt.savefig('normal_test.png')
    plt.close()
    
    # 正态性检验
    k2, p = stats.normaltest(data)
    print(f"正态性检验: D统计量 = {k2:.4f}, p值 = {p:.4f}")
    print(f"分布正态性: {'符合' if p > 0.05 else '不符合'}")
    
    # QQ图
    plt.figure(figsize=(8, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title("Normal Q-Q Plot")
    plt.savefig('normal_qqplot.png')
    plt.close()

# 运行测试
print("测试均匀分布生成器...")
test_uniform(N=10, size=100000)

print("\n测试正态分布生成器...")
test_normal(mean=100, std=15, size=100000)