"""
NumPy 模块5：线性代数
===========================
矩阵乘法、行列式、特征值、求解线性方程组、SVD。
"""

import numpy as np

# ============================================================
# 1. 矩阵乘法
# ============================================================
print("1. 矩阵乘法的三种写法")
print("=" * 60)

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(f"   A:\n{A}")
print(f"   B:\n{B}\n")

# 方式1: np.dot() — 经典
print(f"   np.dot(A, B):\n{np.dot(A, B)}")

# 方式2: @ 运算符 — 最推荐 (Python 3.5+)
print(f"   A @ B:\n{A @ B}")

# 方式3: np.matmul() — 和 @ 等价
print(f"   np.matmul(A, B):\n{np.matmul(A, B)}")

# 注意: A * B 是逐元素乘, 不是矩阵乘!
print(f"   A * B (逐元素乘!):\n{A * B}")
print()

# ============================================================
# 2. 向量点积
# ============================================================
print("2. 向量点积")
print("=" * 60)

v = np.array([1, 2, 3])
w = np.array([4, 5, 6])

print(f"   v = {v}, w = {w}")
print(f"   v @ w 点积:    {v @ w}")                 # 32
print(f"   np.dot(v, w):  {np.dot(v, w)}")
print(f"   np.inner(v,w): {np.inner(v, w)}")        # 内积, 同点积
print()

# ============================================================
# 3. 矩阵转置
# ============================================================
print("3. 矩阵转置")
print("=" * 60)

M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"   M shape {M.shape}:\n{M}")
print(f"   M.T shape {M.T.shape}:\n{M.T}")
print()

# ============================================================
# 4. 矩阵求逆
# ============================================================
print("4. 矩阵求逆")
print("=" * 60)

M = np.array([[4, 7],
              [2, 6]])
M_inv = np.linalg.inv(M)

print(f"   M:\n{M}")
print(f"   M 的逆:\n{M_inv}")
print(f"   M @ M_inv (应 = I):\n{M @ M_inv}")
print()

# ============================================================
# 5. 行列式
# ============================================================
print("5. 行列式")
print("=" * 60)

M = np.array([[4, 7],
              [2, 6]])
det = np.linalg.det(M)
print(f"   det(M) = {det:.2f}")
print(f"   可逆吗? {'是' if det != 0 else '否 (奇异矩阵)'}")
print()

# ============================================================
# 6. 特征值与特征向量
# ============================================================
print("6. 特征值与特征向量")
print("=" * 60)

M = np.array([[4, -2],
              [1,  1]])

eigenvalues, eigenvectors = np.linalg.eig(M)
print(f"   M:\n{M}")
print(f"   特征值: {eigenvalues}")
print(f"   特征向量 (每列一个):\n{eigenvectors}")
print(f"   验证: M @ v1 = {M @ eigenvectors[:, 0]}")
print(f"         lambda1 * v1 = {eigenvalues[0] * eigenvectors[:, 0]}")
print()

# ============================================================
# 7. 求解线性方程组 Ax = b
# ============================================================
print("7. 求解线性方程组")
print("=" * 60)

# 方程组:  2x + y = 5
#          -x + 3y = 8
A = np.array([[2, 1],
              [-1, 3]])
b = np.array([5, 8])

x = np.linalg.solve(A, b)
print(f"   方程组: 2x+y=5, -x+3y=8")
print(f"   解: x = {x[0]:.1f}, y = {x[1]:.1f}")
print(f"   验证 Ax = {A @ x}")
print()

# ============================================================
# 8. 范数 (向量长度 / 矩阵大小)
# ============================================================
print("8. 范数")
print("=" * 60)

v = np.array([3, 4])
print(f"   v = {v}")
print(f"   L2 范数 (欧氏距离):    {np.linalg.norm(v):.2f}")    # 5
print(f"   L1 范数 (曼哈顿距离):  {np.linalg.norm(v, ord=1):.2f}")
print(f"   L∞ 范数 (最大绝对值):  {np.linalg.norm(v, ord=np.inf):.2f}")
print()

# ============================================================
# 9. SVD (奇异值分解)
# ============================================================
print("9. SVD 分解")
print("=" * 60)

M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
U, S, Vt = np.linalg.svd(M)

print(f"   M (3x3):\n{M}\n")
print(f"   U:\n{U}")
print(f"   奇异值 S: {S}")
print(f"   Vt:\n{Vt}")
print(f"   重构 M = U @ diag(S) @ Vt:")
print(f"{U @ np.diag(S) @ Vt[:len(S), :]}")
print()

# ============================================================
# 10. 最小二乘拟合
# ============================================================
print("10. 最小二乘拟合")
print("=" * 60)

# y = mx + b, 找最佳 m 和 b
x_data = np.array([0, 1, 2, 3, 4])
y_data = np.array([1.1, 2.9, 5.2, 6.8, 9.1])

# 构建设计矩阵: [x, 1]
A = np.vstack([x_data, np.ones_like(x_data)]).T
m, b = np.linalg.lstsq(A, y_data, rcond=None)[0]

print(f"   数据点: x={x_data}")
print(f"          y={y_data}")
print(f"   拟合结果: y = {m:.2f}x + {b:.2f}")
print(f"   预测 y(5) = {m * 5 + b:.2f}")

print("\n[OK] NumPy 模块5 完成!")
