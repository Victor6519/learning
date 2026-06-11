"""
Pandas 模块1：Series 与 DataFrame
=====================================
Pandas 两大核心数据结构：Series（一维）和 DataFrame（二维表）。
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. Series — 带标签的一维数组
# ============================================================
print("1. Series — 一维带标签数组")
print("=" * 60)

# 从列表创建
s = pd.Series([10, 20, 30, 40])
print(f"   从列表:\n{s}\n")
print(f"   值: {s.values}")
print(f"   索引: {s.index.tolist()}")

# 自定义索引
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(f"\n   自定义索引:\n{s2}")
print(f"   s2['b'] = {s2['b']}")

# 从字典创建
s3 = pd.Series({'苹果': 5, '香蕉': 3, '橙子': 8})
print(f"\n   从字典:\n{s3}")
print()

# ============================================================
# 2. DataFrame — 二维表 (核心!)
# ============================================================
print("2. DataFrame — 二维表格")
print("=" * 60)

# 从字典创建 (最常用)
df = pd.DataFrame({
    '姓名':   ['张三', '李四', '王五', '赵六'],
    '年龄':   [25, 30, 22, 28],
    '城市':   ['北京', '上海', '广州', '深圳'],
    '工资':   [8000, 12000, 6500, 9500],
})
print(f"   从字典创建:\n{df}\n")

# 从 NumPy 数组创建
df2 = pd.DataFrame(
    np.random.randn(5, 3),
    columns=['A', 'B', 'C'],
    index=['row' + str(i) for i in range(5)]
)
print(f"   从 NumPy:\n{df2}\n")

# 从列表嵌套创建
data = [
    ['AGK', 274.3, -2.36],
    ['ILVFW', 676.9, -4.14],
    ['KRDE', 546.6, -0.68],
]
df3 = pd.DataFrame(data, columns=['序列', '分子量', 'logS'])
print(f"   从列表:\n{df3}\n")

# ============================================================
# 3. DataFrame 核心属性
# ============================================================
print("3. DataFrame 核心属性")
print("=" * 60)

print(f"   shape:    {df.shape}         (行数, 列数)")
print(f"   columns:  {df.columns.tolist()}")
print(f"   index:    {df.index.tolist()}")
print(f"   dtypes:\n{df.dtypes}\n")
print(f"   info():")
df.info()
print()
print(f"   describe():\n{df.describe()}")
print()

# ============================================================
# 4. 快速创建 DataFrame
# ============================================================
print("4. 快速创建方式")
print("=" * 60)

# 从 CSV (字符串模拟)
from io import StringIO
csv_data = StringIO("""
name,age,city
Alice,25,NYC
Bob,30,LA
Charlie,35,SF
""")
df_csv = pd.read_csv(csv_data)
print(f"   read_csv():\n{df_csv}\n")

# 日期范围 + 随机数
dates = pd.date_range('2024-01-01', periods=5, freq='D')
df_date = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=['A', 'B', 'C'])
print(f"   带日期索引:\n{df_date}\n")

# ============================================================
# 5. 列操作
# ============================================================
print("5. 列操作")
print("=" * 60)

# 选一列 → Series
print(f"   选 '年龄' 列:\n{df['年龄']}\n")

# 选多列 → DataFrame
print(f"   选多列:\n{df[['姓名', '工资']]}\n")

# 新增列
df['奖金'] = df['工资'] * 0.1
print(f"   新增 '奖金' 列:\n{df}\n")

# 删除列
df_dropped = df.drop(columns=['城市'])
print(f"   删除 '城市' 列:\n{df_dropped}\n")

print("[OK] Pandas 模块1 完成!")
