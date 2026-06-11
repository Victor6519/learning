"""
PyTorch 模块3：训练循环
===========================
Optimizer、训练/验证循环、学习率、过拟合演示。
"""

import torch
import torch.nn as nn
import numpy as np

# ============================================================
# 1. 生成模拟数据
# ============================================================
print("1. 生成模拟数据")
print("=" * 60)

torch.manual_seed(42)
n_samples = 200

# 二分类: 两类点，线性可分 + 噪声
X = torch.randn(n_samples, 2)
y = ((X[:, 0] + X[:, 1]) > 0).long()   # 对角线分割, 标签 0/1

print(f"   X shape: {X.shape}, y shape: {y.shape}")
print(f"   正样本: {(y==1).sum().item()}, 负样本: {(y==0).sum().item()}")
print()

# ============================================================
# 2. 定义模型、损失函数、优化器
# ============================================================
print("2. 模型 + 损失 + 优化器 (三件套)")
print("=" * 60)

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 2),     # 2类 logits
)

criterion = nn.CrossEntropyLoss()            # 分类损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"   模型: {sum(p.numel() for p in model.parameters())} 个参数")
print(f"   损失函数: CrossEntropyLoss")
print(f"   优化器: Adam (lr=0.01)")
print()

# ============================================================
# 3. 训练循环 (核心!)
# ============================================================
print("3. 训练循环")
print("=" * 60)

n_epochs = 100
for epoch in range(1, n_epochs + 1):
    # ---- 前向传播 ----
    logits = model(X)                         # 预测
    loss = criterion(logits, y)                # 计算损失

    # ---- 反向传播 ----
    optimizer.zero_grad()                      # 清空上次梯度
    loss.backward()                            # 计算梯度
    optimizer.step()                           # 更新参数

    # ---- 日志 ----
    if epoch % 20 == 0:
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        print(f"   Epoch {epoch:3d}: loss={loss.item():.4f}, "
              f"acc={acc.item():.3f}")

print()

# ============================================================
# 4. 完整的训练/验证循环
# ============================================================
print("4. 完整训练/验证流程")
print("=" * 60)

# 重新生成数据 + 手动划分
torch.manual_seed(0)
X = torch.randn(500, 2)
y = ((X[:, 0]**2 + X[:, 1]**2) < 1.0).long()  # 圆形决策边界 (非线性)

# 80/20 划分
n_train = 400
indices = torch.randperm(500)
X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
X_val, y_val = X[indices[n_train:]], y[indices[n_train:]]
print(f"   训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}")

# 更大的模型
model2 = nn.Sequential(
    nn.Linear(2, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 2),
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)

# 训练 + 验证
history = {'train_loss': [], 'val_acc': []}
for epoch in range(1, 201):
    # -- 训练 --
    model2.train()
    logits = model2(X_train)
    loss = criterion(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # -- 验证 (不计算梯度) --
    model2.eval()
    with torch.no_grad():
        val_logits = model2(X_val)
        val_pred = val_logits.argmax(dim=1)
        val_acc = (val_pred == y_val).float().mean()

    history['train_loss'].append(loss.item())
    history['val_acc'].append(val_acc.item())

    if epoch % 50 == 0:
        print(f"   Epoch {epoch:3d}: train_loss={loss.item():.4f}, "
              f"val_acc={val_acc.item():.3f}")

print(f"\n   最终验证准确率: {history['val_acc'][-1]:.3f}")
print()

# ============================================================
# 5. 学习率调度
# ============================================================
print("5. 学习率调度")
print("=" * 60)

model3 = nn.Linear(2, 2)
optimizer = torch.optim.SGD(model3.parameters(), lr=0.1)

# StepLR: 每 step_size 个 epoch 乘以 gamma
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

print(f"   初始 lr: {optimizer.param_groups[0]['lr']}")
for epoch in range(1, 101):
    scheduler.step()
    if epoch % 30 == 0:
        print(f"   Epoch {epoch:3d}: lr = {optimizer.param_groups[0]['lr']:.4f}")
# 100 epoch × 1 step = 100 steps, scheduler steps every 30 epochs


print("\n[OK] PyTorch 模块3 完成!")
