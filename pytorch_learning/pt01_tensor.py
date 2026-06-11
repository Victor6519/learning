"""
PyTorch 模块1：Tensor 基础
==============================
Tensor 创建、属性、运算、GPU 转移、autograd 自动求导。
"""

import torch
import numpy as np

# ============================================================
# 1. Tensor 创建
# ============================================================
print("1. 创建 Tensor")
print("=" * 60)

# 从列表
a = torch.tensor([1, 2, 3])
b = torch.tensor([[1., 2.], [3., 4.]])
print(f"   一维: {a}")
print(f"   二维:\n{b}")
print(f"   类型: {type(a)}")           # torch.Tensor
print()

# 类似 NumPy 的工厂函数
print(f"   zeros(3, 4):\n{torch.zeros(3, 4)}")
print(f"   ones(2, 3):\n{torch.ones(2, 3)}")
print(f"   eye(3):\n{torch.eye(3)}")
print(f"   arange(0, 10, 2): {torch.arange(0, 10, 2)}")
print(f"   linspace(0, 1, 5): {torch.linspace(0, 1, 5)}")
print(f"   rand(2, 3):\n{torch.rand(2, 3)}")
print(f"   randn(2, 3):\n{torch.randn(2, 3)}")
print()

# 从 NumPy 互转
np_arr = np.array([1., 2., 3.])
tensor = torch.from_numpy(np_arr)
back_to_np = tensor.numpy()
print(f"   NumPy -> Tensor: {tensor}, dtype={tensor.dtype}")
print(f"   Tensor -> NumPy: {back_to_np}, type={type(back_to_np)}")
print(f"   共享内存? {np_arr[0] == tensor[0]}")  # True, 同一块内存
print()

# ============================================================
# 2. Tensor 属性
# ============================================================
print("2. Tensor 属性")
print("=" * 60)

t = torch.randn(2, 3, 4)
print(f"   shape:    {t.shape}         (类似 ndim)")
print(f"   dtype:    {t.dtype}         (数据类型)")
print(f"   device:   {t.device}        (CPU/GPU)")
print(f"   numel():  {t.numel()}       (总元素数)")
print(f"   requires_grad: {t.requires_grad}  (是否追踪梯度)")
print()

# ============================================================
# 3. 基本运算 (与 NumPy 高度一致)
# ============================================================
print("3. 基本运算")
print("=" * 60)

a = torch.tensor([1., 2., 3., 4.])
b = torch.tensor([10., 20., 30., 40.])

print(f"   a + b   = {a + b}")
print(f"   a * b   = {a * b}         (逐元素乘)")
print(f"   a ** 2  = {a ** 2}")
print(f"   a > 2   = {a > 2}")
print(f"   sin(a)  = {torch.sin(a)}")

# 矩阵乘法: @ 或 torch.mm / torch.matmul
A = torch.randn(2, 3)
B = torch.randn(3, 4)
print(f"\n   A(2,3) @ B(3,4) shape: {(A @ B).shape}")
print(f"   torch.mm 只支持2D: {torch.mm(A, B).shape}")
print()

# ============================================================
# 4. 索引切片 (和 NumPy 一样)
# ============================================================
print("4. 索引切片")
print("=" * 60)

t = torch.arange(12).reshape(3, 4)
print(f"   原始:\n{t}")
print(f"   t[0, 0]  = {t[0, 0]}")
print(f"   t[:, 0]  = {t[:, 0]}")
print(f"   t[1:, 1:3]:\n{t[1:, 1:3]}")
print(f"   t[t > 5]  = {t[t > 5]}    (布尔索引)")
print()

# ============================================================
# 5. GPU 转移
# ============================================================
print("5. GPU / CPU 转移")
print("=" * 60)

cpu_tensor = torch.randn(1000, 1000)
print(f"   CPU: {cpu_tensor.device}")

if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to('cuda')
    print(f"   GPU: {gpu_tensor.device}")
    # GPU 运算后再移回
    result = (gpu_tensor @ gpu_tensor).to('cpu')
    print(f"   回 CPU: {result.device}")
else:
    print("   (无 GPU, 跳过)")
print()

# ============================================================
# 6. Autograd — 自动求导 (PyTorch 的灵魂)
# ============================================================
print("6. Autograd 自动求导")
print("=" * 60)

# requires_grad=True 告诉 PyTorch 追踪这个 tensor 的所有操作
x = torch.tensor([2., 3.], requires_grad=True)
print(f"   x = {x}")

# y = x[0]^2 + 3*x[1]^3
y = x[0] ** 2 + 3 * x[1] ** 3
print(f"   y = x[0]^2 + 3*x[1]^3 = {y.item():.2f}")

# 反向传播: 计算 dy/dx = [2*x0, 9*x1^2] = [4, 81]
y.backward()
print(f"   dy/dx = {x.grad}")           # [4, 81]
print(f"   手动验证: [2*2, 9*3^2] = [4, 81]")

# ---- 计算图示例 ----
print(f"\n   简单神经网络的前向+反向:")
w = torch.tensor([1.5], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)
x = torch.tensor([2.0])
target = torch.tensor([1.0])

# 前向: y_pred = w*x + b, loss = (y_pred - target)^2
y_pred = w * x + b
loss = (y_pred - target) ** 2

print(f"     y_pred = {y_pred.item():.2f}, loss = {loss.item():.2f}")

# 反向: 求 dloss/dw, dloss/db
loss.backward()
print(f"     dloss/dw = {w.grad.item():.1f}")     # 2*(w*x+b-target)*x = 2*(3+0.5-1)*2 = 10
print(f"     dloss/db = {b.grad.item():.1f}")     # 2*(w*x+b-target)*1 = 2*(3+0.5-1)*1 = 5
print()

print("[OK] PyTorch 模块1 完成!")
