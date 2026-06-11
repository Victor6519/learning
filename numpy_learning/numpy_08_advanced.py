"""
NumPy 模块8：高级操作
===========================
排序、去重、masked array、内存布局、实用技巧。
"""

import numpy as np

# ============================================================
# 1. 排序
# ============================================================
print("1. 排序")
print("=" * 60)

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"   原始: {arr}")

# np.sort 返回副本
print(f"   np.sort() 升序:  {np.sort(arr)}")
print(f"   arr[::-1] 降序:  {np.sort(arr)[::-1]}")

# arr.sort 原地排序
arr_copy = arr.copy()
arr_copy.sort()
print(f"   .sort() 原地:   {arr_copy}")

# argsort: 返回排序后的索引
print(f"   argsort 索引:  {np.argsort(arr)}")
print(f"   用索引取值:    {arr[np.argsort(arr)]}")
print()

# ---- 二维按行/列排序 ----
arr2d = np.array([[3, 1, 4],
                  [1, 5, 9],
                  [2, 6, 5]])
print(f"   二维:\n{arr2d}")
print(f"   sort axis=0 (每列排序):\n{np.sort(arr2d, axis=0)}")
print(f"   sort axis=1 (每行排序):\n{np.sort(arr2d, axis=1)}")
print()

# ============================================================
# 2. 去重
# ============================================================
print("2. 去重")
print("=" * 60)

arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
print(f"   原始: {arr}")
print(f"   np.unique(): {np.unique(arr)}")
print(f"   唯一值 + 计数: {np.unique(arr, return_counts=True)}")
print(f"   唯一值 + 首次出现位置: {np.unique(arr, return_index=True)}")
print()

# ============================================================
# 3. 掩码数组 (masked array)
# ============================================================
print("3. 掩码数组 — 处理缺失/无效值")
print("=" * 60)

# 数据中有无效值 (-999 代表缺失)
data = np.array([1.2, 2.5, -999, 3.1, -999, 4.0])
masked = np.ma.masked_equal(data, -999)

print(f"   原始: {data}")
print(f"   掩码后: {masked}")
print(f"   掩码 (True=被遮蔽): {masked.mask}")
print(f"   忽略 -999 的均值: {masked.mean():.2f}")
print(f"   对比原始均值:     {data.mean():.2f}  (被 -999 拉低了)")
print()

# ============================================================
# 4. 边界处理
# ============================================================
print("4. clip — 限制值范围")
print("=" * 60)

arr = np.array([-2, 5, 8, 12, 3, 20])
clipped = np.clip(arr, 0, 10)
print(f"   原始:   {arr}")
print(f"   clip(0, 10): {clipped}")
print()

# ============================================================
# 5. 处理 NaN 和 inf
# ============================================================
print("5. NaN 处理")
print("=" * 60)

arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
print(f"   原始: {arr}")
print(f"   isnan:  {np.isnan(arr)}")
print(f"   无 NaN 的均值: {np.nanmean(arr):.2f}")
print(f"   无 NaN 的和:   {np.nansum(arr):.2f}")
print(f"   无 NaN 的标准差:{np.nanstd(arr):.2f}")

# 填充 NaN
filled = np.nan_to_num(arr, nan=0.0)
print(f"   nan_to_num(填0): {filled}")
print()

# ============================================================
# 6. 向量化条件函数
# ============================================================
print("6. np.select / np.piecewise — 多条件向量化")
print("=" * 60)

scores = np.array([45, 67, 82, 91, 55, 73, 88, 95])

# 多条件映射: 不及格/及格/良好/优秀
conditions = [scores < 60, (scores >= 60) & (scores < 80),
              (scores >= 80) & (scores < 90), scores >= 90]
choices = ["不及格", "及格", "良好", "优秀"]
grades = np.select(conditions, choices, default="未知")

print(f"   分数: {scores}")
print(f"   等级: {grades}")

# 另一种写法: np.piecewise (数学函数分段)
x = np.array([-2, -1, 0, 1, 2])
result = np.piecewise(x, [x < 0, x >= 0], [lambda x: x**2, lambda x: x**3])
print(f"   分段: x<0取平方, x>=0取立方: {result}")
print()

# ============================================================
# 7. 内存布局与性能优化
# ============================================================
print("7. 内存布局优化")
print("=" * 60)

# 大数组优先预分配
n = 100000

# 慢: 每次追加都重新分配内存
import time
t0 = time.time()
result = np.array([])
for i in range(1000):
    result = np.append(result, i)
t1 = time.time()

# 快: 先用 list 收集再转 array
t2 = time.time()
items = []
for i in range(1000):
    items.append(i)
result2 = np.array(items)
t3 = time.time()

print(f"   np.append 逐次追加: {t1-t0:.4f}s")
print(f"   list→array 批量:    {t3-t2:.4f}s")
print()

# ============================================================
# 8. 用 einsum 做多维运算
# ============================================================
print("8. einsum (爱因斯坦求和约定)")
print("=" * 60)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"   A:\n{A}")
print(f"   B:\n{B}")

# 矩阵乘法: ij,jk->ik
print(f"   矩阵乘 (ij,jk->ik):\n{np.einsum('ij,jk->ik', A, B)}")

# 逐元素乘后求和 (点积的推广)
print(f"   逐元素乘后求和 (ij,ij->): {np.einsum('ij,ij->', A, B)}")

# 对角线求和 (迹)
print(f"   trace (ii->): {np.einsum('ii->', A)}")

# 外积
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
print(f"   外积 (i,j->ij):\n{np.einsum('i,j->ij', a, b)}")
print()

# ============================================================
# 9. 文件 IO
# ============================================================
print("9. 文件读写")
print("=" * 60)

import os

filepath = os.path.join(os.path.dirname(__file__), "numpy_demo.npy")

# 保存
data = np.random.randn(100, 5)
np.save(filepath, data)
print(f"   已保存 {data.shape} 到 numpy_demo.npy")

# 加载
loaded = np.load(filepath)
print(f"   已加载: shape={loaded.shape}")

# 文本格式
np.savetxt(filepath.replace(".npy", ".csv"), data[:5], delimiter=",", fmt="%.4f")
print(f"   已保存前5行到 numpy_demo.csv")

os.remove(filepath)
os.remove(filepath.replace(".npy", ".csv"))

print("\n[OK] NumPy 模块8 完成!")
print("\n=== NumPy 8 个模块全部完成! ===")
