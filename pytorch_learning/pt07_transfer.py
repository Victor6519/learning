"""
PyTorch 模块7：迁移学习 & 预训练模型
=========================================
加载预训练模型、冻结/微调、特征提取、自定义分类头。
"""

import torch
import torch.nn as nn

# ============================================================
# 1. 加载预训练模型
# ============================================================
print("1. 加载预训练模型")
print("=" * 60)

# ResNet18 — 在 ImageNet 上预训练
try:
    from torchvision import models
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    print(f"   已加载 ResNet18 (IMAGENET 预训练)")
except:
    # 如果没有 torchvision 或网络不通, 用随机权重演示
    resnet = models.resnet18(weights=None)
    print(f"   已加载 ResNet18 (随机权重, 演示用)")

print(f"   最后一层: {resnet.fc}")
print(f"   总参数: {sum(p.numel() for p in resnet.parameters()):,}")
print()

# ============================================================
# 2. 冻结特征提取层
# ============================================================
print("2. 冻结预训练权重")
print("=" * 60)

# 冻结除了 fc 层以外的所有参数
for name, param in resnet.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in resnet.parameters() if not p.requires_grad)
print(f"   冻结: {frozen:,}, 可训练: {trainable:,}")
print()

# ============================================================
# 3. 替换分类头
# ============================================================
print("3. 替换分类头")
print("=" * 60)

# ResNet18 原始 fc: 512 → 1000 (ImageNet有1000类)
# 改成自己的任务: 比如 5 类分类
num_classes = 5
resnet.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes),
)
print(f"   新分类头: 512 → 256 → 5")
print(f"   可训练参数: {sum(p.numel() for p in resnet.parameters() if p.requires_grad):,}")
print()

# ============================================================
# 4. 分阶段解冻 (渐进式微调)
# ============================================================
print("4. 渐进式微调策略")
print("=" * 60)

resnet2 = models.resnet18(weights=None)

# 阶段1: 只训练分类头 (前面全冻)
# 阶段2: 解冻 layer4
# 阶段3: 解冻 layer3
# 阶段4: 全部解冻 (学习率极低)

stages = [
    ("阶段1: 只训练fc", ['fc']),
    ("阶段2: +layer4",  ['fc', 'layer4']),
    ("阶段3: +layer3",  ['fc', 'layer4', 'layer3']),
    ("阶段4: 全部",     None),   # None = 全部
]

for desc, layers in stages:
    # 冻结全部
    for p in resnet2.parameters():
        p.requires_grad = False

    # 解冻指定层
    if layers is None:
        for p in resnet2.parameters():
            p.requires_grad = True
    else:
        for name, param in resnet2.named_parameters():
            if any(l in name for l in layers):
                param.requires_grad = True

    trainable = sum(p.numel() for p in resnet2.parameters() if p.requires_grad)
    print(f"   {desc}: 可训练 {trainable:,} 参数")

print()

# ============================================================
# 5. 用预训练模型做特征提取 (不需要分类头)
# ============================================================
print("5. 特征提取器 — 去掉最后一层")
print("=" * 60)

# 去掉 fc 和 avgpool, 只留卷积部分做特征提取
feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
# ResNet 结构: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
# 去掉最后两层 (avgpool+fc), 保留卷积特征

x = torch.randn(1, 3, 224, 224)
features = feature_extractor(x)
print(f"   输入: {x.shape}")
print(f"   卷积特征: {features.shape}   (512通道, 7×7空间)")
print(f"   用途: 提取图像embedding, 用于检索/聚类/度量学习")
print()

# ============================================================
# 6. 实用技巧速查
# ============================================================
print("6. 微调实用技巧")
print("=" * 60)
print("""
   学习率: 预训练层用 1e-4, 新分类头用 1e-3 (10倍差)
   数据增强: 微调时强增强 (RandAugment), 防止小数据集过拟合
   早停:    val_loss 不降 5 epoch 就停
   选择模型: 分类用 ResNet/EfficientNet, 分割用 UNet, NLP用 BERT
   判别式学习率: 越靠近输出的层学习率越高
""")

print("[OK] PyTorch 模块7 完成!")
