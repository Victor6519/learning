"""
PyTorch 模块2：神经网络基础
==============================
nn.Module, nn.Linear, 激活函数, 前向传播, 参数管理。
"""

import torch
import torch.nn as nn

# ============================================================
# 1. nn.Linear — 全连接层 (y = Wx + b)
# ============================================================
print("1. nn.Linear — 全连接层")
print("=" * 60)

# in_features=4, out_features=3
linear = nn.Linear(4, 3)    # 4个输入 → 3个输出
print(f"   层: {linear}")
print(f"   权重 W shape: {linear.weight.shape}")   # (3, 4)
print(f"   偏置 b shape: {linear.bias.shape}")     # (3,)

# 前向传播
x = torch.randn(2, 4)       # batch=2, features=4
y = linear(x)               # → shape (2, 3)
print(f"   输入 (2,4) → 输出 {y.shape}")
print(f"   y:\n{y}")
print()

# ============================================================
# 2. 常用激活函数
# ============================================================
print("2. 激活函数")
print("=" * 60)

x = torch.linspace(-3, 3, 7)
print(f"   x = {x}\n")

# ReLU: max(0, x) — 最常用
print(f"   ReLU(x):      {torch.relu(x)}")
# Sigmoid: 1/(1+e^-x) — 二分类输出
print(f"   Sigmoid(x):   {torch.sigmoid(x)}")
# Tanh: (e^x-e^-x)/(e^x+e^-x) — RNN常用
print(f"   Tanh(x):      {torch.tanh(x)}")
# Softmax: e^x/sum(e^x) — 多分类输出
print(f"   Softmax(x):   {torch.softmax(x, dim=0)}")
print()

# ============================================================
# 3. nn.Module — 构建自定义网络
# ============================================================
print("3. 自定义网络 (nn.Module)")
print("=" * 60)


class SimpleMLP(nn.Module):
    """一个简单的多层感知器"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # 定义层
        self.fc1 = nn.Linear(in_dim, hidden_dim)    # 输入→隐藏
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 隐藏→隐藏
        self.fc3 = nn.Linear(hidden_dim, out_dim)    # 隐藏→输出
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)               # 20% dropout

    def forward(self, x):
        # 定义前向传播流程
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)               # 输出层不加激活
        return x


model = SimpleMLP(in_dim=10, hidden_dim=32, out_dim=2)
print(f"   模型:\n{model}")

# 查看参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n   总参数量: {total_params}")
print(f"   可训练参数: {trainable_params}")

# 遍历参数名
print(f"\n   各层参数:")
for name, param in model.named_parameters():
    print(f"     {name:20s} shape={str(param.shape):15s} "
          f"requires_grad={param.requires_grad}")
print()

# 前向传播
x = torch.randn(4, 10)    # batch=4, dim=10
y = model(x)
print(f"   输入 {x.shape} → 输出 {y.shape}")

# ============================================================
# 4. nn.Sequential — 快速搭建
# ============================================================
print("\n\n4. nn.Sequential — 链式搭建")
print("=" * 60)

seq_model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)
print(f"   Sequential模型:\n{seq_model}")
print(f"   输出: {seq_model(x).shape}")
print()

# ============================================================
# 5. 损失函数速览
# ============================================================
print("5. 常用损失函数")
print("=" * 60)

pred = torch.tensor([[2.0, 1.0, 0.5]])    # 预测 logits (batch=1, 3类)
target_cls = torch.tensor([0])              # 真实类别

# CrossEntropyLoss: 分类任务标配 (内含softmax)
ce_loss = nn.CrossEntropyLoss()
print(f"   CrossEntropyLoss: {ce_loss(pred, target_cls).item():.4f}")

# MSELoss: 回归任务
pred_reg = torch.tensor([2.5, 1.0, 3.0])
target_reg = torch.tensor([2.0, 1.5, 2.5])
mse_loss = nn.MSELoss()
print(f"   MSELoss:          {mse_loss(pred_reg, target_reg).item():.4f}")

# BCELoss: 二分类
bce_loss = nn.BCEWithLogitsLoss()
pred_bin = torch.tensor([0.8, -0.3])
target_bin = torch.tensor([1.0, 0.0])
print(f"   BCEWithLogitsLoss: {bce_loss(pred_bin, target_bin).item():.4f}")
print()

print("[OK] PyTorch 模块2 完成!")
