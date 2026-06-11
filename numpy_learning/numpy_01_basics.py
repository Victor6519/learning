"""
NumPy 模块1：数组创建与基本操作
===================================
创建数组、数组属性、数据类型、基本运算。
"""

import numpy as np

# ============================================================
# 1. 从 Python 列表创建数组
# ============================================================
print("1. 从 list 创建数组")
print("=" * 60)

# np.array() — 最常用的创建方式
a = np.array([1, 2, 3, 4, 5])                 # 一维
b = np.array([[1, 2, 3], [4, 5, 6]])           # 二维 (矩阵)

print(f"   一维数组: {a}")
print(f"   二维数组:\n{b}")
print(f"   a 的类型: {type(a)}")                # <class 'numpy.ndarray'>
print()

# ============================================================
# 2. 数组核心属性
# ============================================================
print("2. 数组核心属性")
print("=" * 60)

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print(f"   数组:\n{arr}\n")
print(f"   ndim    维度数:    {arr.ndim}")       # 2
print(f"   shape   形状:      {arr.shape}")       # (3, 4)
print(f"   size    总元素数:  {arr.size}")        # 12
print(f"   dtype   元素类型:  {arr.dtype}")       # int32/int64
print(f"   itemsize 每元素字节: {arr.itemsize}")  # 8
print()

# ============================================================
# 3. 内置快速创建函数
# ============================================================
print("3. 快速创建数组的函数")
print("=" * 60)

# zeros / ones — 全0 / 全1
print(f"   zeros(5):         {np.zeros(5)}")
print(f"   ones(3, 4):\n{np.ones((3, 4))}")

# arange — 等差数列 (类似 range但返回数组)
print(f"   arange(0, 10, 2):  {np.arange(0, 10, 2)}")

# linspace — 指定数量的等差数列 (含终点)
print(f"   linspace(0, 1, 5): {np.linspace(0, 1, 5)}")

# eye / identity — 单位矩阵
print(f"   eye(3):\n{np.eye(3)}")

# full — 填充指定值
print(f"   full((2, 3), 7):\n{np.full((2, 3), 7)}")

# empty — 只分配内存不初始化 (随机垃圾值, 但更快)
print(f"   empty(3):  {np.empty(3)}")
print()

# ============================================================
# 4. 数据类型 (dtype)
# ============================================================
print("4. 数据类型")
print("=" * 60)

print(f"   int64:     {np.array([1, 2, 3]).dtype}")           # 默认整数
print(f"   float64:   {np.array([1.0, 2.0, 3.0]).dtype}")    # 默认浮点
print(f"   指定 int32: {np.array([1, 2, 3], dtype=np.int32).dtype}")
print(f"   指定 float32: {np.array([1, 2, 3], dtype=np.float32).dtype}")
print(f"   布尔型:     {np.array([True, False, True]).dtype}")
print(f"   复数:       {np.array([1+2j, 3+4j]).dtype}")
print(f"   字符串:     {np.array(['hello', 'world']).dtype}")

# 类型转换: .astype()
arr_int = np.array([1, 2, 3])
arr_float = arr_int.astype(np.float64)
print(f"   类型转换 int->float: {arr_float}  dtype={arr_float.dtype}")
print()

# ============================================================
# 5. 数组基本运算 (逐元素)
# ============================================================
print("5. 基本运算 (逐元素)")
print("=" * 60)

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(f"   a = {a}")
print(f"   b = {b}")
print(f"   a + b     = {a + b}")           # 逐元素加
print(f"   a - b     = {a - b}")           # 逐元素减
print(f"   a * b     = {a * b}")           # 逐元素乘 (不是矩阵乘法!)
print(f"   a / b     = {a / b}")           # 逐元素除
print(f"   a ** 2    = {a ** 2}")          # 逐元素平方
print(f"   a > 2     = {a > 2}")           # 逐元素比较 → 布尔数组
print(f"   a % 2     = {a % 2}")           # 逐元素取余
print(f"   np.sin(a) = {np.sin(a)}")       # 逐元素 sin
print(f"   np.log(a) = {np.log(a)}")       # 逐元素 log
print()

# ============================================================
# 6. 数组与标量的运算 (广播)
# ============================================================
print("6. 数组与标量运算")
print("=" * 60)

print(f"   a + 10  = {a + 10}")            # 每个元素 +10
print(f"   a * 2   = {a * 2}")             # 每个元素 *2
print(f"   a / 2   = {a / 2}")             # 每个元素 /2
print()

# ============================================================
# 7. 数组拷贝
# ============================================================
print("7. 数组拷贝 (重要!)")
print("=" * 60)

arr = np.array([1, 2, 3])

# 浅拷贝 (=) : 两个变量指向同一块内存
view = arr
view[0] = 999
print(f"   浅拷贝修改后 arr = {arr}  (arr 也被改了!)")

# 深拷贝 (.copy()) : 复制一份新数据
arr = np.array([1, 2, 3])
copy = arr.copy()
copy[0] = 999
print(f"   深拷贝修改后 arr = {arr}  (arr 不受影响)")
print()

print("[OK] NumPy 模块1 完成!")
