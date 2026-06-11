"""
NumPy 模块6：随机数
=========================
随机数生成、概率分布、随机抽样、随机种子。
"""

import numpy as np

# ============================================================
# 1. 随机种子 — 可复现性
# ============================================================
print("1. 随机种子")
print("=" * 60)

np.random.seed(42)
print(f"   seed(42) 第1次: {np.random.rand(3)}")
np.random.seed(42)
print(f"   seed(42) 第2次: {np.random.rand(3)}  (和上面一模一样)")
print(f"   不设 seed 的话每次都不一样")
print()

# ============================================================
# 2. 均匀分布
# ============================================================
print("2. 均匀分布")
print("=" * 60)

# rand: [0, 1) 均匀分布
print(f"   rand(5) [0,1):      {np.random.rand(5)}")
print(f"   rand(3, 2) 二维:\n{np.random.rand(3, 2)}")

# uniform: 指定范围的均匀分布
print(f"   uniform(10, 20, 5) [10,20): {np.random.uniform(10, 20, 5)}")

# randint: 随机整数
print(f"   randint(0, 100, 10):      {np.random.randint(0, 100, 10)}")
print()

# ============================================================
# 3. 正态分布
# ============================================================
print("3. 正态分布 (高斯分布)")
print("=" * 60)

# randn: 标准正态 N(0, 1)
print(f"   randn(5) ~ N(0,1): {np.random.randn(5)}")

# normal: 指定均值和标准差
heights = np.random.normal(loc=170, scale=7, size=1000)
print(f"   normal(loc=170, scale=7, size=1000)")
print(f"     模拟身高: 均值={heights.mean():.1f}, 标准差={heights.std():.1f}")
print()

# ============================================================
# 4. 常用离散分布
# ============================================================
print("4. 离散分布")
print("=" * 60)

# binomial: 二项分布 (n 次伯努利实验成功次数)
# 抛 10 次硬币, 重复 1000 组
coins = np.random.binomial(n=10, p=0.5, size=1000)
print(f"   抛10次硬币, 正面朝上次数 (重复1000组):")
print(f"     均值 (预期 np=5): {coins.mean():.2f}")
print(f"     最小/最大: {coins.min()}/{coins.max()}")

# poisson: 泊松分布
# 平均 λ=3 的罕见事件计次
events = np.random.poisson(lam=3, size=1000)
print(f"   泊松 λ=3: 均值={events.mean():.2f}")
print()

# ============================================================
# 5. 随机洗牌与抽样
# ============================================================
print("5. 随机洗牌与抽样")
print("=" * 60)

arr = np.arange(10)
print(f"   原始: {arr}")

# shuffle: 原地打乱
np.random.shuffle(arr)
print(f"   shuffle 后: {arr}")

# permutation: 返回打乱的副本
arr = np.arange(10)
perm = np.random.permutation(arr)
print(f"   permutation: {perm}")

# choice: 随机抽样 (可重复/不重复)
samples = np.random.choice(arr, size=5, replace=False)
print(f"   choice(size=5, 不放回): {samples}")

# 带权抽样 (p 必须 sum=1)
p = [0.02]*5 + [0.18]*5    # 前5个低权重, 后5个高权重, sum=1.0
weighted = np.random.choice(arr, size=5, p=p)
print(f"   choice(加权): {weighted}")
print()

# ============================================================
# 6. 随机子集分割 (train/test split)
# ============================================================
print("6. 应用: 数据集划分")
print("=" * 60)

np.random.seed(0)
n_samples = 10
indices = np.random.permutation(n_samples)
split = int(0.7 * n_samples)

train_idx = indices[:split]
test_idx = indices[split:]

print(f"   总样本: {n_samples}")
print(f"   训练集索引 (70%): {sorted(train_idx)}")
print(f"   测试集索引 (30%): {sorted(test_idx)}")
print()

# ============================================================
# 7. 生成特定分布的随机数据
# ============================================================
print("7. 更多分布速查")
print("=" * 60)

print(f"   beta(2, 5) 贝塔分布:       {np.random.beta(2, 5, 3)}")
print(f"   gamma(2, 3) 伽马分布:      {np.random.gamma(2, 3, 3)}")
print(f"   exponential(1) 指数分布:    {np.random.exponential(1, 3)}")
print(f"   logistic(0, 1) 逻辑分布:   {np.random.logistic(0, 1, 3)}")
print(f"   chisquare(3) 卡方分布:     {np.random.chisquare(3, 3)}")

print("\n[OK] NumPy 模块6 完成!")
