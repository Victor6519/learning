"""
PyTorch 模块5：CNN 卷积神经网络
====================================
Conv2d, MaxPool2d, 特征图可视化, 简单图像分类。
"""

import torch
import torch.nn as nn

# ============================================================
# 1. Conv2d — 卷积层
# ============================================================
print("1. Conv2d — 二维卷积")
print("=" * 60)

# 模拟一张 28x28 灰度图 (batch=1, channel=1, H=28, W=28)
image = torch.randn(1, 1, 28, 28)

# in_channels=1 (灰度), out_channels=8 (8个卷积核), kernel_size=3
conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
out = conv1(image)

print(f"   输入 shape: {image.shape}   (batch, C, H, W)")
print(f"   卷积核: 1→8 通道, 3x3, padding=1")
print(f"   输出 shape: {out.shape}     (H,W不变因为padding=1)")
print(f"   参数量: {sum(p.numel() for p in conv1.parameters())}")
# 参数: 8个核 × (1×3×3 + 1偏置) = 80
print()

# ============================================================
# 2. MaxPool2d — 池化(下采样)
# ============================================================
print("2. MaxPool2d — 最大池化")
print("=" * 60)

pool = nn.MaxPool2d(kernel_size=2, stride=2)
pooled = pool(out)
print(f"   池化前: {out.shape}")
print(f"   池化后: {pooled.shape}    (H,W 减半)")
print(f"   操作: 每 2x2 窗口取最大值, 步长=2")
print()

# ============================================================
# 3. 构建一个简单的 CNN
# ============================================================
print("3. 构建 CNN — MNIST风格")
print("=" * 60)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 卷积部分: 逐层压缩空间, 增加通道
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # 28×28 → 28×28×16
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 28×28 → 14×14
            nn.Conv2d(16, 32, 3, padding=1),  # 14×14 → 14×14×32
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 14×14 → 7×7
        )
        # 全连接部分: 把卷积特征映射到类别
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                      # 7×7×32 → 1568
            nn.Linear(7 * 7 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


model = SimpleCNN(num_classes=10)
x = torch.randn(4, 1, 28, 28)    # 4张 28x28 灰度图
y = model(x)
print(f"   输入: {x.shape}")
print(f"   输出: {y.shape}    (4张图 → 10类的logits)")

# 参数量分析
conv_params = sum(p.numel() for p in model.conv_layers.parameters())
fc_params = sum(p.numel() for p in model.fc_layers.parameters())
print(f"   卷积参数: {conv_params}, 全连接参数: {fc_params}")
print(f"   总参数: {conv_params + fc_params}")
print()

# ============================================================
# 4. 特征图可视化 (中间层输出)
# ============================================================
print("4. 中间层输出 (特征图)")
print("=" * 60)

# 只看卷积部分的前向
with torch.no_grad():
    x = torch.randn(1, 1, 28, 28)
    c1 = model.conv_layers[0](x)    # Conv1 输出
    print(f"   Conv1 后: {c1.shape}   (16个特征图, 28×28)")
    print(f"     特征图1 激活范围: [{c1[0,0].min():.2f}, {c1[0,0].max():.2f}]")

    p1 = model.conv_layers[2](c1)   # MaxPool1
    print(f"   MaxPool1 后: {p1.shape}   (16个特征图, 14×14)")

    c2 = model.conv_layers[3](p1)   # Conv2 输出
    print(f"   Conv2 后: {c2.shape}   (32个特征图, 14×14)")
    print(f"     通道数增多, 空间尺寸减小 — CNN的典型模式")
print()

# ============================================================
# 5. 1x1 卷积 — 通道压缩
# ============================================================
print("5. 1x1 卷积 — 降维/升维")
print("=" * 60)

feat = torch.randn(1, 64, 14, 14)   # 64通道特征图
compress = nn.Conv2d(64, 16, kernel_size=1)
reduced = compress(feat)
print(f"   64通道 → 16通道 (1x1卷积)")
print(f"   输入: {feat.shape}, 输出: {reduced.shape}")
print(f"   参数量仅: {sum(p.numel() for p in compress.parameters())}   (64×16+16)")
print(f"   用途: 降维/升维, 不改变空间尺寸, 极省参数")
print()

print("[OK] PyTorch 模块5 完成!")
