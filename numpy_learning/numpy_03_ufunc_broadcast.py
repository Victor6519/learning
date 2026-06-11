"""
NumPy 模块3：通用函数 (ufunc) 与广播 (broadcasting)
=========================================================
向量化计算、常用数学函数、广播规则。
"""

import numpy as np

# ============================================================
# 1. 为什么用 ufunc 而不是 Python 循环
# ============================================================
print("1. 向量化 vs 循环 — 速度差距")
print("=" * 60)

# Python 循环
arr = np.arange(10000000)   # 1000万

import time
t0 = time.perf_counter()
result_py = [x ** 2 + 2 * x + 1 for x in arr]
t1 = time.perf_counter()

# NumPy 向量化
t2 = time.perf_counter()
result_np = arr ** 2 + 2 * arr + 1
t3 = time.perf_counter()

py_time = t1 - t0
np_time = max(t3 - t2, 1e-9)  # 防止除0
print(f"   Python 循环: {py_time:.4f}s")
print(f"   NumPy 向量化: {np_time:.4f}s")
print(f"   速度比: {py_time/np_time:.0f}x")
print()

# ============================================================
# 2. 常用一元 ufunc
# ============================================================
print("2. 常用一元 ufunc (对每个元素操作)")
print("=" * 60)

arr = np.array([0, 1, 2, 3, 4, 5])
print(f"   原始: {arr}\n")

print(f"   np.sqrt(x)    平方根:    {np.sqrt(arr)}")
print(f"   np.square(x)  平方:      {np.square(arr)}")
print(f"   np.exp(x)     e^x:       {np.exp(arr[:4])}")
print(f"   np.log(x)     ln(x):     {np.log(arr[1:4])}")
print(f"   np.log10(x)   log10:     {np.log10(arr[1:5])}")
print(f"   np.abs(x)     绝对值:    {np.abs(np.array([-3, 5, -1, 0]))}")
print(f"   np.ceil(x)    向上取整:  {np.ceil(np.array([1.2, 3.7, -0.3]))}")
print(f"   np.floor(x)   向下取整:  {np.floor(np.array([1.2, 3.7, -0.3]))}")
print(f"   np.round(x)   四舍五入:  {np.round(np.array([1.2, 3.7, 2.5]))}")
print(f"   np.sign(x)    符号:      {np.sign(np.array([-3, 0, 5]))}")
print(f"   np.isnan(x)   是否 NaN:  {np.isnan(np.array([1, np.nan, 3]))}")
print()

# ============================================================
# 3. 常用二元 ufunc
# ============================================================
print("3. 常用二元 ufunc")
print("=" * 60)

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(f"   a = {a}, b = {b}\n")
print(f"   np.add(a, b)       加法:       {np.add(a, b)}")
print(f"   np.subtract(a, b)  减法:       {np.subtract(a, b)}")
print(f"   np.multiply(a, b)  乘法:       {np.multiply(a, b)}")
print(f"   np.divide(a, b)    除法:       {np.divide(a, b)}")
print(f"   np.power(a, 2)     幂:         {np.power(a, 2)}")
print(f"   np.maximum(a, b)   逐元素取最大: {np.maximum(a, b)}")
print(f"   np.minimum(a, b)   逐元素取最小: {np.minimum(a, b)}")
print()

# ============================================================
# 4. 广播 (Broadcasting) — NumPy 最核心的机制
# ============================================================
print("4. 广播机制")
print("=" * 60)

# 规则: 从最后一维向前比较, 维度相同或其中一个为1才能广播
# 广播 = 自动扩展小数组的形状来匹配大数组

# 例1: 标量 + 数组 (最简广播)
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(f"   原数组 (2,3):\n{arr}")
print(f"   arr + 10 (标量广播):\n{arr + 10}\n")

# 例2: 行 + 列 — shape (3,) 广播到 (2, 3)
row = np.array([10, 20, 30])  # shape (3,)
print(f"   row (3,): {row}")
print(f"   arr + row:\n{arr + row}")     # row 被复制到每行

# 例3: 列 + 行 — shape (2,1) 广播到 (2, 3)
col = np.array([[100], [200]])  # shape (2,1)
print(f"\n   col (2,1):\n{col}")
print(f"   arr + col:\n{arr + col}")     # col 被复制到每列
print()

# 例4: 广播不兼容的情况
print("   广播不兼容示例:")
print(f"   (2, 3) + (2,)  -> 不行, 最后维度 3 != 2")
print(f"   (2, 3) + (3,)  -> 可以, 3==3")
print(f"   (2, 3) + (2, 1)-> 可以, 3 可以广播")
print(f"   (2, 3) + (1, 3)-> 可以, 2 可以广播")
print()

# ============================================================
# 5. 广播的实际应用
# ============================================================
print("5. 广播实际应用")
print("=" * 60)

# 5a. 数据标准化 (每列减去均值)
data = np.array([[1, 100, 10],
                 [3, 200, 20],
                 [5, 300, 30]])
print(f"   原始数据:\n{data}")
mean = data.mean(axis=0)   # 每列的均值 shape (3,)
print(f"   每列均值: {mean}")
print(f"   去均值后:\n{data - mean}\n")    # (3,3) - (3,) 广播

# 5b. 外积 — 两个向量的所有组合
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
outer = a[:, np.newaxis] * b  # (3,1) * (3,) → (3,3)
print(f"   外积 (a * b):\n  a={a}, b={b}")
print(f"  a[:, None] * b:\n{outer}")

# 5c. 距离矩阵 — 常用技巧
points = np.array([[0, 0],
                   [3, 0],
                   [0, 4]])    # 3 个二维点
# 用广播计算所有点对之间的欧氏距离
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
# diff shape: (3, 1, 2) - (1, 3, 2) → (3, 3, 2)
distances = np.sqrt(np.sum(diff ** 2, axis=2))
print(f"\n   三点距离矩阵:\n{distances}")
print()

print("[OK] NumPy 模块3 完成!")
