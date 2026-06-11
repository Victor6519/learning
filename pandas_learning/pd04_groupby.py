"""
Pandas 模块4：分组聚合 (groupby)
====================================
split-apply-combine：分组→计算→合并。
"""

import pandas as pd
import numpy as np

# 模拟数据集：多肽在不同溶剂中的溶解度
np.random.seed(42)
df = pd.DataFrame({
    '多肽':   ['AGK', 'ILVFW', 'KRDE', 'AGK', 'ILVFW', 'KRDE',
               'AGK', 'ILVFW', 'KRDE', 'AGK', 'ILVFW', 'KRDE'],
    '溶剂':   ['水', '水', '水', 'DMF', 'DMF', 'DMF',
               'DCM', 'DCM', 'DCM', 'DMSO', 'DMSO', 'DMSO'],
    'logS':   [-2.3, -4.1, -0.7, -1.8, -3.6, -0.5,
               -1.5, -3.2, -1.0, -2.0, -3.8, -0.6],
    '温度_C': [25, 25, 25, 25, 25, 25, 30, 30, 30, 30, 30, 30],
})
print(f"数据:\n{df}\n")

# ============================================================
# 1. groupby 基础 — split-apply-combine
# ============================================================
print("1. groupby 基础")
print("=" * 60)

# 按多肽分组, 算 logS 均值
grouped = df.groupby('多肽')
print(f"   按多肽分组 (GroupBy对象): {type(grouped).__name__}")
print(f"   分组数: {grouped.ngroups}")
print(f"   groups:")
for name, grp in grouped:
    print(f"     {name}: {grp['logS'].tolist()}")

print(f"\n   每组的 logS 均值:\n{grouped['logS'].mean()}\n")

# ============================================================
# 2. 多列分组
# ============================================================
print("2. 多列分组")
print("=" * 60)

# 按多肽 + 溶剂分组
result = df.groupby(['多肽', '溶剂'])['logS'].mean()
print(f"   多列分组 mean:\n{result}\n")

# 转成宽表
result_wide = result.unstack()
print(f"   转宽表 (unstack):\n{result_wide}\n")

# ============================================================
# 3. 常用聚合函数
# ============================================================
print("3. 常用聚合函数")
print("=" * 60)

# 单个聚合
print(f"   sum():\n{df.groupby('多肽')['logS'].sum()}\n")

# 多个聚合 — agg()
stats = df.groupby('多肽')['logS'].agg(['mean', 'std', 'min', 'max', 'count'])
print(f"   agg(['mean','std','min','max','count']):\n{stats}\n")

# 对不同列用不同聚合
multi = df.groupby('多肽').agg({
    'logS': 'mean',
    '温度_C': 'max',
    '溶剂': 'count',   # 计数
})
print(f"   对不同列不同聚合:\n{multi}\n")

# ============================================================
# 4. transform — 保持原形状
# ============================================================
print("4. transform — 广播回原形状")
print("=" * 60)

# agg 是压缩, transform 是广播
df['组内均值'] = df.groupby('多肽')['logS'].transform('mean')
df['组内排名'] = df.groupby('多肽')['logS'].transform('rank')
print(f"   transform 结果:\n{df[['多肽','溶剂','logS','组内均值','组内排名']]}\n")

# ============================================================
# 5. filter — 按组过滤
# ============================================================
print("5. filter — 过滤整个组")
print("=" * 60)

# 只保留均值 < -1.5 的组
filtered = df.groupby('多肽').filter(lambda g: g['logS'].mean() < -1.5)
print(f"   filter 均值<-1.5 的组:\n{filtered[['多肽','溶剂','logS']]}\n")

# ============================================================
# 6. pivot_table — 数据透视表
# ============================================================
print("6. pivot_table — 透视表")
print("=" * 60)

pivot = df.pivot_table(
    values='logS',
    index='多肽',
    columns='溶剂',
    aggfunc='mean'
)
print(f"   透视表:\n{pivot}\n")

# 带汇总
pivot2 = df.pivot_table(
    values='logS',
    index='多肽',
    columns='溶剂',
    aggfunc='mean',
    margins=True,          # 加 all 汇总行/列
)
print(f"   带汇总:\n{pivot2}\n")

print("[OK] Pandas 模块4 完成!")
