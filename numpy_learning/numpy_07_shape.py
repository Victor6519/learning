"""
NumPy 模块7：形状操作
===========================
reshape, 展平, 拼接, 分割, 维度扩展与压缩。
"""

import numpy as np

# ============================================================
# 1. reshape — 改变形状
# ============================================================
print("1. reshape — 改变形状")
print("=" * 60)

arr = np.arange(12)
print(f"   原始 (12,): {arr}")

# reshape 返回视图 (view)
print(f"   reshape(3, 4):\n{arr.reshape(3, 4)}")
print(f"   reshape(2, 6):\n{arr.reshape(2, 6)}")
print(f"   reshape(2, 3, 2):\n{arr.reshape(2, 3, 2)}")

# -1 = 自动推导该维度
print(f"   reshape(-1, 4):\n{arr.reshape(-1, 4)}")   # 自动算 12/4=3
print(f"   reshape(2, -1):\n{arr.reshape(2, -1)}")   # 自动算 12/2=6
print()

# ============================================================
# 2. 展平 (flatten vs ravel)
# ============================================================
print("2. 展平的三种方式")
print("=" * 60)

arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])
print(f"   原始 (2,3):\n{arr2d}\n")

# flatten: 返回副本 (新数组)
flat = arr2d.flatten()
flat[0] = 999
print(f"   flatten() 改值后原数组不变 (副本):")
print(f"     新数组: {flat}")
print(f"     原数组: {arr2d}")

# ravel: 返回视图 (与原数组共享内存)
raveled = arr2d.ravel()
raveled[0] = 999
print(f"   ravel() 改值后原数组变了 (视图!):")
print(f"     raveled: {raveled}")
print(f"     原数组: {arr2d}")

# reshape(-1): 也是视图
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
reshaped = arr2d.reshape(-1)
reshaped[0] = 999
print(f"   reshape(-1) 改值后原数组变了: {arr2d}")
print()

# ============================================================
# 3. 拼接
# ============================================================
print("3. 数组拼接")
print("=" * 60)

a = np.array([[1, 2],
              [3, 4]])
000000000000000000
b = np.array([[5, 6],
              [7, 8]])

print(f"   a:\n{a}")
print(f"   b:\n{b}\n")

# vstack: 垂直堆叠 (行方向)
print(f"   vstack (垂直):\n{np.vstack([a, b])}")

# hstack: 水平堆叠 (列方向)
print(f"   hstack (水平):\n{np.hstack([a, b])}")

# concatenate: 通用拼接
print(f"   concatenate axis=0:\n{np.concatenate([a, b], axis=0)}")
print(f"   concatenate axis=1:\n{np.concatenate([a, b], axis=1)}")

# column_stack: 把一维数组当列堆
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print(f"   column_stack: 合并 {x} 和 {y} 为列:\n{np.column_stack([x, y])}")

# r_ / c_: 快速拼接简写
print(f"   r_[a, b] (行拼接):\n{np.r_[a, b]}")
print(f"   c_[a, b] (列拼接):\n{np.c_[a, b]}")
print()

# ============================================================
# 4. 分割
# ============================================================
print("4. 数组分割")
print("=" * 60)

arr = np.arange(12).reshape(3, 4)
print(f"   原始 (3,4):\n{arr}\n")

# vsplit: 垂直分割
top, middle, bottom = np.vsplit(arr, 3)
print(f"   vsplit(3) 第一块:\n{top}")

# hsplit: 水平分割
left, right = np.hsplit(arr, 2)
print(f"   hsplit(2) 左半:\n{left}")
print(f"   hsplit(2) 右半:\n{right}")
print()

# ============================================================
# 5. 维度扩展
# ============================================================
print("5. 维度扩展")
print("=" * 60)

arr = np.array([1, 2, 3])   # shape (3,)
print(f"   原始 shape: {arr.shape}")

# newaxis / None: 增加一维
print(f"   行向量 shape: {arr[np.newaxis, :].shape}   → {arr[np.newaxis, :]}")
print(f"   列向量 shape: {arr[:, np.newaxis].shape}\n{arr[:, np.newaxis]}")

# expand_dims
print(f"   expand_dims(arr, 0) shape: {np.expand_dims(arr, 0).shape}")
print(f"   expand_dims(arr, 1) shape: {np.expand_dims(arr, 1).shape}")
print()

# ============================================================
# 6. 压缩维度
# ============================================================
print("6. squeeze — 删除长度为1的维度")
print("=" * 60)

arr = np.array([[[1, 2, 3]]])  # shape (1, 1, 3)
print(f"   原始 shape: {arr.shape}")
print(f"   squeeze shape: {np.squeeze(arr).shape}")
print(f"   squeeze 结果: {np.squeeze(arr)}")
print()

# ============================================================
# 7. 转置高维数组
# ============================================================
print("7. transpose — 交换维度")
print("=" * 60)

arr = np.arange(24).reshape(2, 3, 4)  # (2, 3, 4)
print(f"   原始 shape: {arr.shape}")

# 交换轴
transposed = arr.transpose(2, 0, 1)   # (4, 2, 3)
print(f"   transpose(2, 0, 1) shape: {transposed.shape}")

# T 属性 (适用于二维矩阵)
M = np.array([[1, 2, 3], [4, 5, 6]])
print(f"   M.T shape: {M.T.shape}")
print()

# ============================================================
# 8. tile — 重复拼接
# ============================================================
print("8. tile / repeat — 重复")
print("=" * 60)

arr = np.array([1, 2, 3])
print(f"   原始: {arr}")
print(f"   np.tile(arr, 3)       重复整个:  {np.tile(arr, 3)}")
print(f"   np.tile(arr, (2, 3))  2行3列:\n{np.tile(arr, (2, 3))}")
print(f"   np.repeat(arr, 3)     每元素重复: {np.repeat(arr, 3)}")
print()

print("[OK] NumPy 模块7 完成!")
