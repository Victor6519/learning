"""
Pandas 模块7：apply / map / 向量化
=====================================
行列级操作, 自定义函数, 性能对比。
"""

import pandas as pd
import numpy as np
import time

df = pd.DataFrame({
    '序列':   ['AGK', 'ILVFW', 'KRDE', 'AKEFTL'],
    'logS':  [-2.36, -4.14, -0.68, -2.80],
    'MW':    [274.3, 676.9, 546.6, 707.8],
})
print(f"数据:\n{df}\n")

# ============================================================
# 1. map — Series 元素级映射
# ============================================================
print("1. map — 单列映射")
print("=" * 60)

# 字典映射
aa_map = {'AGK': '三肽', 'ILVFW': '五肽', 'KRDE': '四肽', 'AKEFTL': '六肽'}
df['类型'] = df['序列'].map(aa_map)
print(f"   map 字典:\n{df[['序列','类型']]}\n")

# 函数映射
df['abs_logS'] = df['logS'].map(abs)
print(f"   map abs:\n{df[['logS','abs_logS']]}\n")

# 分类映射
df['sol_class'] = df['logS'].map(lambda x: 'high' if x > -2 else 'low')
print(f"   map lambda:\n{df[['logS','sol_class']]}\n")

# ============================================================
# 2. apply — 行/列级函数
# ============================================================
print("2. apply — 行列操作")
print("=" * 60)

# 对列 apply (axis=0, 默认): 每列单独处理
print(f"   apply(np.mean):\n{df[['logS','MW']].apply(np.mean)}\n")

# 对行 apply (axis=1): 每行单独处理
def describe_row(row):
    return f"{row['序列']}: logS={row['logS']:.1f}, MW={row['MW']:.0f}"

df['描述'] = df.apply(describe_row, axis=1)
print(f"   apply(axis=1) 行操作:\n{df[['描述']]}\n")

# 多列组合计算
df['logS_per_MW'] = df.apply(
    lambda row: row['logS'] / row['MW'] * 100, axis=1
)
print(f"   apply 多列组合 (logS/MW*100):\n{df[['序列','logS_per_MW']]}\n")

# ============================================================
# 3. applymap — DataFrame 逐元素
# ============================================================
print("3. applymap — 逐元素操作")
print("=" * 60)

num_df = df[['logS', 'MW']]
print(f"   applymap 取四位小数:\n{num_df.applymap(lambda x: f'{x:.4f}')}\n")

# ============================================================
# 4. 向量化 > apply > map 的性能对比
# ============================================================
print("4. 性能对比")
print("=" * 60)

big = pd.DataFrame({'A': np.random.randn(100000), 'B': np.random.randn(100000)})

# 向量化 (直接列运算)
t0 = time.perf_counter()
big['sum_vec'] = big['A'] + big['B']
t1 = time.perf_counter()

# apply
t2 = time.perf_counter()
big['sum_apply'] = big.apply(lambda r: r['A'] + r['B'], axis=1)
t3 = time.perf_counter()

print(f"   向量化: {(t1-t0)*1000:.1f}ms")
print(f"   apply:   {(t3-t2)*1000:.1f}ms")
print(f"   推荐: 能用向量化就别用 apply")
print()

# ============================================================
# 5. pipe — 链式操作
# ============================================================
print("5. pipe — 链式调用")
print("=" * 60)


def add_features(df):
    df = df.copy()
    df['MW_sqrt'] = np.sqrt(df['MW'])
    df['logS_round'] = df['logS'].round(1)
    return df


def filter_big(df, threshold=500):
    return df[df['MW'] > threshold]


# pipe 把函数串起来, 避免中间变量
result = (df.pipe(add_features)
          .pipe(filter_big, threshold=500)
          .pipe(lambda d: d[['序列', 'MW', 'MW_sqrt', 'logS_round']]))

print(f"   pipe 链式:\n{result}\n")

print("[OK] Pandas 模块7 完成!")
