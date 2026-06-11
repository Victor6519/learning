"""
NumPy 模块4：聚合与统计
=============================
求和、均值、标准差、最大最小、按轴计算、累积操作。
"""

import numpy as np

# 生成模拟数据: 3 次实验, 每实验 5 个观测值
np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=(3, 5))
# 3行 = 3个实验组, 5列 = 每组5个样本

print("   模拟数据 (3组实验, 每组5个值):")
print(f"{data}\n")
print(f"   shape = {data.shape}  (行=组, 列=样本)")
print()

# ============================================================
# 1. 全局聚合 (不分轴, 返回标量)
# ============================================================
print("1. 全局聚合 (整个数组)")
print("=" * 60)

print(f"   np.sum(arr)      总和:   {np.sum(data):.2f}")
print(f"   np.mean(arr)     均值:   {np.mean(data):.2f}")
print(f"   np.std(arr)      标准差: {np.std(data):.2f}")
print(f"   np.var(arr)      方差:   {np.var(data):.2f}")
print(f"   np.min(arr)      最小值: {np.min(data):.2f}")
print(f"   np.max(arr)      最大值: {np.max(data):.2f}")
print(f"   np.median(arr)   中位数: {np.median(data):.2f}")
print(f"   np.percentile(arr, 25)  第25百分位: {np.percentile(data, 25):.2f}")
print(f"   np.percentile(arr, 75)  第75百分位: {np.percentile(data, 75):.2f}")
print()

# 同时也有方法形式:
print(f"   arr.sum():  {data.sum():.2f}  (效果相同)")
print(f"   arr.mean(): {data.mean():.2f}")
print()

# ============================================================
# 2. 按轴聚合 (axis=0 按列, axis=1 按行)
# ============================================================
print("2. 按轴聚合 (axis 参数)")
print("=" * 60)



print(f"   原始 (3组, 5样本):\n{data}\n")

# axis=0: 沿行方向/跨行 (对每组同一位置的样本做统计)
print(f"   每组均值   axis=0: {data.mean(axis=0)}")  # 5个数,每列均值
print(f"   每组总和   axis=0: {data.sum(axis=0)}")

# axis=1: 沿列方向/跨列 (对每组内5个样本做统计)
print(f"\n   每实验均值  axis=1: {data.mean(axis=1)}") # 3个数,每行均值
print(f"   每实验总和  axis=1: {data.sum(axis=1)}")
print(f"   每实验标准差 axis=1: {data.std(axis=1)}")
print()

# ---- axis 记忆口诀 ----
print("   axis 记忆: axis=0 → 压缩行(垂直), axis=1 → 压缩列(水平)")
print(f"   shape {data.shape} → mean(axis=0) shape = {data.mean(axis=0).shape}")
print(f"   shape {data.shape} → mean(axis=1) shape = {data.mean(axis=1).shape}")
print()

# ============================================================
# 3. 带条件的聚合
# ============================================================
print("3. 带条件聚合")
print("=" * 60)

print(f"   data > 100 的元素数:    {np.sum(data > 100)}")
print(f"   data > 100 的比例:      {np.mean(data > 100):.2%}")
print(f"   data > 100 的元素求和:   {np.sum(data[data > 100]):.2f}")
print(f"   data 中 NaN 的个数:      {np.sum(np.isnan(data))}")
print()

# ============================================================
# 4. 最大值/最小值的索引
# ============================================================
print("4. argmax / argmin — 取极值位置")
print("=" * 60)

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"   数组: {arr}")
print(f"   argmax (最大值位置): {np.argmax(arr)}  → 值为 {arr[np.argmax(arr)]}")
print(f"   argmin (最小值位置): {np.argmin(arr)}  → 值为 {arr[np.argmin(arr)]}")

# 二维 — 返回的是展平后的索引
arr2d = np.array([[1, 5, 3],
                  [4, 2, 6]])
print(f"\n   二维:\n{arr2d}")
print(f"   argmax 展平索引: {np.argmax(arr2d)}")
print(f"   转为行列: {np.unravel_index(np.argmax(arr2d), arr2d.shape)}")
print()

# ============================================================
# 5. 累积操作
# ============================================================
print("5. 累积操作")
print("=" * 60)

arr = np.array([1, 2, 3, 4, 5])
print(f"   原始:        {arr}")
print(f"   cumsum 累积和: {np.cumsum(arr)}")
print(f"   cumprod 累积积: {np.cumprod(arr)}")
print()

# ============================================================
# 6. 差分
# ============================================================
print("6. 差分 (diff)")
print("=" * 60)

arr = np.array([1, 3, 6, 10, 15])
print(f"   原始: {arr}")
print(f"   一阶差分: {np.diff(arr)}   (3-1, 6-3, 10-6, 15-10)")
print(f"   二阶差分: {np.diff(arr, n=2)}")
print()

# ============================================================
# 7. 综合示例: 快速数据分析
# ============================================================
print("7. 综合示例: 快速描述性统计")
print("=" * 60)

# 模拟 100 个学生的成绩
np.random.seed(0)
scores = np.random.normal(loc=72, scale=12, size=100)
scores = np.clip(scores, 0, 100)   # 限制在 0~100

print(f"   成绩分析 (100 人):")
print(f"     平均分:   {scores.mean():.1f}")
print(f"     中位数:   {np.median(scores):.1f}")
print(f"     标准差:   {scores.std():.1f}")
print(f"     最高分:   {scores.max():.0f}")
print(f"     最低分:   {scores.min():.0f}")
print(f"     四分位数: Q1={np.percentile(scores,25):.0f}, "
      f"Q2={np.percentile(scores,50):.0f}, "
      f"Q3={np.percentile(scores,75):.0f}")
print(f"     及格率:   {np.mean(scores >= 60):.1%}")
print(f"     优秀率:   {np.mean(scores >= 90):.1%}")
print(f"     不及格人数: {np.sum(scores < 60)}")

print("\n[OK] NumPy 模块4 完成!")
