"""
PyTorch 模块4：数据加载
===========================
Dataset, DataLoader, 批处理, 数据增强, 自定义数据集。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

# ============================================================
# 1. TensorDataset — 最简方式
# ============================================================
print("1. TensorDataset — 从 Tensor 直接创建")
print("=" * 60)

X = torch.randn(100, 5)
y = torch.randn(100, 1)

# 把 X 和 y 打包成一个数据集
dataset = TensorDataset(X, y)
print(f"   数据集大小: {len(dataset)}")
print(f"   dataset[0] = (张量shape={dataset[0][0].shape}, 张量shape={dataset[0][1].shape})")
print()

# ============================================================
# 2. DataLoader — 批处理 + 打乱
# ============================================================
print("2. DataLoader — 批处理")
print("=" * 60)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"   每批 16 个样本, 共 {len(loader)} 批")
for batch_idx, (batch_x, batch_y) in enumerate(loader):
    print(f"   批 {batch_idx}: X shape={batch_x.shape}, y shape={batch_y.shape}")
    if batch_idx >= 2:
        print(f"   ...")
        break
print()

# ============================================================
# 3. 自定义 Dataset
# ============================================================
print("3. 自定义 Dataset")
print("=" * 60)


class PeptideDataset(Dataset):
    """
    自定义数据集: 多肽序列 + 溶解度标签。

    用法:
        dataset = PeptideDataset(sequences, solubilities)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
    """
    def __init__(self, sequences, labels, transform=None):
        """
        sequences: list of str (氨基酸序列)
        labels:    list/np.array of float (溶解度 logS)
        transform: 可选, 数据增强/预处理函数
        """
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

        # 简单的氨基酸→数字编码: 20种AA映射到0-19
        self.aa_to_idx = {aa: i for i, aa in enumerate(
            'ACDEFGHIKLMNPQRSTVWY')}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # 序列 → 独热编码张量 (length × 20)
        encoded = torch.zeros(len(seq), 20)
        for i, aa in enumerate(seq):
            if aa in self.aa_to_idx:
                encoded[i, self.aa_to_idx[aa]] = 1.0

        # 可选的数据增强
        if self.transform:
            encoded = self.transform(encoded)

        return encoded, label


# 模拟数据 (统一长度, 方便默认 collate 批处理)
seqs = [
    'AGK', 'ILV', 'KRD', 'AKE', 'AGA',
    'SLA', 'YYY', 'GGG', 'ALA', 'FLF',
] * 10   # 100条, 每条约 3 残基
labels = np.random.uniform(-6, 0, 100)

dataset = PeptideDataset(seqs, labels)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

print(f"   数据集: {len(dataset)} 条")
for batch_x, batch_y in loader:
    print(f"   一批: X shape={batch_x.shape}  (batch, seq_len, 20), "
          f"y shape={batch_y.shape}")
    break

print()

# ============================================================
# 4. collate_fn — 处理变长序列
# ============================================================
print("4. collate_fn — 处理变长序列")
print("=" * 60)


def pad_collate(batch):
    """
    将变长序列填充到同一长度。
    batch: list of (encoded_seq, label)
    """
    sequences, labels = zip(*batch)
    # 填充到本批次最大长度
    lengths = torch.tensor([len(s) for s in sequences])
    max_len = lengths.max()

    padded = torch.zeros(len(sequences), max_len, 20)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq

    labels = torch.tensor(labels)
    return padded, labels, lengths


# 变长序列数据集
var_seqs = ['A', 'AGK', 'ILVFW', 'AKEFTL', 'AGAGAGAGAG',
            'SLAFQLLEQLIQQR', 'AAA', 'GG', 'VVVVV', 'KKKKK'] * 6
var_labels = np.random.uniform(-6, 0, 60)
var_dataset = PeptideDataset(var_seqs, var_labels)
var_loader = DataLoader(var_dataset, batch_size=8, shuffle=True,
                        collate_fn=pad_collate)

for padded, labels, lengths in var_loader:
    print(f"   填充后: padded shape={padded.shape}, "
          f"lengths={lengths.tolist()}")
    break

print()

# ============================================================
# 5. 数据划分 (train/val/test)
# ============================================================
print("5. 数据划分")
print("=" * 60)


def split_dataset(dataset, ratios=(0.7, 0.15, 0.15)):
    """将数据集按比例分割"""
    from torch.utils.data import random_split
    n = len(dataset)
    splits = [int(n * r) for r in ratios]
    splits[-1] = n - sum(splits[:-1])   # 补齐余数
    return random_split(dataset, splits)


full_ds = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
train_ds, val_ds, test_ds = split_dataset(full_ds)

print(f"   总: {len(full_ds)}, 训练: {len(train_ds)}, "
      f"验证: {len(val_ds)}, 测试: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

print(f"   训练批数: {len(train_loader)}")
print()

print("[OK] PyTorch 模块4 完成!")
