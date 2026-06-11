"""
Pandas 模块6：缺失值处理
=============================
NaN 检测、删除、填充、插值。
"""

import pandas as pd
import numpy as np

# 构建含缺失值的数据
df = pd.DataFrame({
    '序列':   ['AGK', 'ILVFW', None, 'AKEFTL', 'KRDE'],
    'logS':   [-2.36, np.nan, -0.68, -2.80, None],
    '溶剂':   ['水', 'DMF', np.nan, 'DMSO', '水'],
    'MW':     [274.3, 676.9, 546.6, None, 546.6],
})
print(f"原始数据:\n{df}\n")

# ============================================================
# 1. 检测缺失值
# ============================================================
print("1. 检测缺失值")
print("=" * 60)

print(f"   isnull():\n{df.isnull()}\n")
print(f"   每列缺失数:\n{df.isnull().sum()}\n")
print(f"   总缺失数: {df.isnull().sum().sum()}")
print(f"   非缺失数:\n{df.notnull().sum()}\n")

# ============================================================
# 2. 删除缺失值
# ============================================================
print("2. 删除缺失值")
print("=" * 60)

# 删除含 NaN 的行
print(f"   dropna() 删除含 NaN 的行:\n{df.dropna()}\n")

# 删除含 NaN 的列
print(f"   dropna(axis=1) 删除含 NaN 的列:\n{df.dropna(axis=1)}\n")

# thresh: 至少保留 N 个非 NaN 的行
print(f"   dropna(thresh=3) 至少3个非NaN:\n{df.dropna(thresh=3)}\n")

# ============================================================
# 3. 填充缺失值
# ============================================================
print("3. 填充缺失值")
print("=" * 60)

# 用固定值填充
print(f"   fillna(0):\n{df['logS'].fillna(0)}\n")

# 用均值填充
print(f"   fillna(mean):\n{df['logS'].fillna(df['logS'].mean())}\n")

# 用前一个值填充 (forward fill)
print(f"   ffill():\n{df.fillna(method='ffill')}\n")

# 用后一个值填充 (backward fill)
print(f"   bfill():\n{df.fillna(method='bfill')}\n")

# 多列不同填充值
df_filled = df.copy()
df_filled['序列'] = df_filled['序列'].fillna('UNKNOWN')
df_filled['logS'] = df_filled['logS'].fillna(df['logS'].median())
df_filled['溶剂'] = df_filled['溶剂'].fillna('水')
df_filled['MW'] = df_filled['MW'].fillna(df['MW'].mean())
print(f"   多列不同填充:\n{df_filled}\n")

# ============================================================
# 4. 插值
# ============================================================
print("4. 插值 (interpolate)")
print("=" * 60)

# 时间序列插值
ts = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0])
print(f"   原始: {ts.tolist()}")
print(f"   linear 插值: {ts.interpolate().tolist()}")
print(f"   quadratic 插值: {ts.interpolate(method='quadratic').tolist()}")
print()

# ============================================================
# 5. 替换值
# ============================================================
print("5. replace — 替换特定值")
print("=" * 60)

df2 = pd.DataFrame({
    'score': [85, -999, 92, -999, 78, 65],  # -999 代表缺失
})
print(f"   原始: {df2['score'].tolist()}")
print(f"   replace(-999, NaN): {df2['score'].replace(-999, np.nan).tolist()}")

# 多值替换
df2['grade'] = ['A', 'B', 'C', 'A', 'B', 'C']
mapped = df2['grade'].replace({'A': '优良', 'B': '中等', 'C': '及格'})
print(f"   replace 映射: {mapped.tolist()}")

print("\n[OK] Pandas 模块6 完成!")
