"""
Pandas 模块8：排序、位移、去重、窗口函数
=============================================
sort_values, shift, rank, rolling, 实战数据清洗。
"""

import pandas as pd
import numpy as np

# 模拟实验数据
df = pd.DataFrame({
    '多肽':   ['AGK', 'AGK', 'ILVFW', 'ILVFW', 'KRDE', 'KRDE', 'AKEFTL', 'AKEFTL'],
    '溶剂':   ['水', 'DMF', '水', 'DMF', '水', 'DMF', '水', 'DMF'],
    'logS':   [-2.3, -1.8, -4.1, -3.6, -0.7, -0.5, -3.0, -2.5],
    '日期':   pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-01',
                               '2024-01-02', '2024-01-01', '2024-01-03',
                               '2024-01-03', '2024-01-04']),
})

# ============================================================
# 1. sort_values — 排序
# ============================================================
print("1. sort_values — 排序")
print("=" * 60)

print(f"   按 logS 升序:\n{df.sort_values('logS')}\n")
print(f"   按 logS 降序:\n{df.sort_values('logS', ascending=False)}\n")
print(f"   多列排序 (溶剂升序, logS降序):\n"
      f"{df.sort_values(['溶剂', 'logS'], ascending=[True, False])}\n")

# ============================================================
# 2. rank — 排名
# ============================================================
print("2. rank — 排名")
print("=" * 60)

df['logS_rank'] = df['logS'].rank(ascending=False)  # logS 越高越好
print(f"   排名:\n{df[['多肽','logS','logS_rank']]}\n")

# 组内排名
df['组内排名'] = df.groupby('溶剂')['logS'].rank(ascending=False)
print(f"   组内排名 (按溶剂分组):\n{df[['多肽','溶剂','logS','组内排名']]}\n")

# ============================================================
# 3. shift — 位移 (算变化量)
# ============================================================
print("3. shift — 前置/后置")
print("=" * 60)

ts = pd.Series([10, 20, 30, 40, 50])
print(f"   原始: {ts.tolist()}")
print(f"   shift(1):  {ts.shift(1).tolist()}    (前一行)")
print(f"   shift(-1): {ts.shift(-1).tolist()}   (后一行)")
print(f"   diff():    {ts.diff().tolist()}      (差值 = 当前-shift1)")

# 实战: 按多肽算 logS 变化
df_sorted = df.sort_values(['多肽', '日期'])
df_sorted['prev_logS'] = df_sorted.groupby('多肽')['logS'].shift(1)
df_sorted['change'] = df_sorted['logS'] - df_sorted['prev_logS']
print(f"\n   多肽 logS 变化:\n{df_sorted[['多肽','日期','logS','prev_logS','change']]}\n")

# ============================================================
# 4. 去重
# ============================================================
print("4. drop_duplicates — 去重")
print("=" * 60)

dup_data = pd.DataFrame({
    'seq': ['AGK', 'AGK', 'AGK', 'ILVFW', 'ILVFW'],
    'logS': [2.3, 2.3, 2.5, 4.1, 4.1],
})
print(f"   原始:\n{dup_data}\n")
print(f"   drop_duplicates():\n{dup_data.drop_duplicates()}\n")
print(f"   按 seq 去重 (保留第一个):\n"
      f"{dup_data.drop_duplicates(subset='seq', keep='first')}\n")

# ============================================================
# 5. rolling — 滑动窗口
# ============================================================
print("5. rolling — 滑动窗口")
print("=" * 60)

scores = pd.Series([85, 62, 91, 47, 78, 53, 95, 70])
print(f"   成绩: {scores.tolist()}")
print(f"   rolling(3).mean():  {scores.rolling(3).mean().round(2).tolist()}")
print(f"   rolling(3).std():   {scores.rolling(3).std().round(1).tolist()}")
print(f"   expanding().mean(): {scores.expanding().mean().round(1).tolist()}")

# 实战: 滑动窗口预测多肽溶解度趋势
df_time = df.sort_values('日期')
df_time['rolling_logS'] = df_time.groupby('溶剂')['logS'].transform(
    lambda x: x.rolling(2, min_periods=1).mean()
)
print(f"\n   按溶剂滑动均值:\n{df_time[['溶剂','日期','logS','rolling_logS']]}\n")

# ============================================================
# 6. 常用小技巧
# ============================================================
print("6. 常用小技巧")
print("=" * 60)

# nunique — 唯一值数量
print(f"   nunique (唯一值数): {df['多肽'].nunique()} 种多肽")

# value_counts — 值计数
print(f"   value_counts:\n{df['溶剂'].value_counts()}\n")

# sample — 随机抽样
print(f"   sample(3):\n{df.sample(3, random_state=42)}\n")

# head / tail
print(f"   head(2):\n{df.head(2)}")
print(f"   tail(2):\n{df.tail(2)}")

print("\n[OK] Pandas 模块8 完成!")
print("\n=== Pandas 8 个模块全部完成! ===")
