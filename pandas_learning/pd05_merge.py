"""
Pandas 模块5：合并与连接
=============================
merge, join, concat — DataFrame 的拼接操作。
"""

import pandas as pd
import numpy as np

# 示例数据
peptides = pd.DataFrame({
    '序列':   ['AGK', 'ILVFW', 'KRDE', 'AKEFTL'],
    '分子量': [274.3, 676.9, 546.6, 707.8],
    '残基数': [3, 5, 4, 6],
})
solubility = pd.DataFrame({
    '序列':   ['AGK', 'ILVFW', 'KRDE', 'XXX'],
    'logS':  [-2.36, -4.14, -0.68, -1.0],
    '溶剂':   ['水', '水', '水', 'DMF'],
})
print(f"多肽:\n{peptides}\n")
print(f"溶解度:\n{solubility}\n")

# ============================================================
# 1. merge — 类似 SQL JOIN
# ============================================================
print("1. merge — SQL 风格连接")
print("=" * 60)

# inner join: 只保留两边都有的
print(f"   inner join (默认):\n"
      f"{pd.merge(peptides, solubility, on='序列')}\n")

# left join: 保留左边所有行
print(f"   left join:\n"
      f"{pd.merge(peptides, solubility, on='序列', how='left')}\n")

# outer join: 保留两边所有行
print(f"   outer join:\n"
      f"{pd.merge(peptides, solubility, on='序列', how='outer')}\n")

# ============================================================
# 2. merge 不同列名
# ============================================================
print("2. merge — 不同列名")
print("=" * 60)

sol2 = solubility.rename(columns={'序列': 'code'})
print(f"   列名不同: left_on / right_on")
print(f"{pd.merge(peptides, sol2, left_on='序列', right_on='code', how='inner')}\n")

# ============================================================
# 3. concat — 纵向/横向拼接
# ============================================================
print("3. concat — 堆叠")
print("=" * 60)

new_data = pd.DataFrame({
    '序列': ['YYY', 'GGG'],
    '分子量': [507.5, 189.2],
    '残基数': [3, 3],
})

# 纵向堆叠 (axis=0, 默认)
stacked = pd.concat([peptides, new_data], axis=0, ignore_index=True)
print(f"   纵向堆叠 (追加行):\n{stacked}\n")

# 横向堆叠 (axis=1)
extra = pd.DataFrame({
    'pI': [10.1, 5.9, 7.0, 6.5],
    'GRAVY': [-0.83, 2.88, -3.85, 0.05],
})
hstacked = pd.concat([peptides, extra], axis=1)
print(f"   横向堆叠 (追加列):\n{hstacked}\n")

# ============================================================
# 4. join — 按索引连接
# ============================================================
print("4. join — 按索引连接")
print("=" * 60)

df1 = peptides.set_index('序列')
df2 = extra
df2.index = ['AGK', 'ILVFW', 'KRDE', 'AKEFTL']

joined = df1.join(df2)
print(f"   按索引 join:\n{joined}\n")

# ============================================================
# 5. merge 四种方式速查
# ============================================================
print("5. merge 方式对比")
print("=" * 60)

print("""
   how='inner'  取交集 (两边都有的键才保留)
   how='left'   左表全保留, 右表匹配不上的填NaN
   how='right'  右表全保留
   how='outer'  取并集 (两边都保留, 缺的填NaN)

   concat(axis=0)  纵向追加行
   concat(axis=1)  横向追加列
   join()          按索引连接 (类似 merge on index)
""")

print("[OK] Pandas 模块5 完成!")
