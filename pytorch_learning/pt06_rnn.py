"""
PyTorch 模块6：RNN / LSTM 序列建模
=========================================
RNN, LSTM, GRU, 序列分类, 多肽溶解度回归。
"""

import torch
import torch.nn as nn
import numpy as np

# ============================================================
# 1. RNN 基础
# ============================================================
print("1. RNN — 循环神经网络")
print("=" * 60)

# input_size=10 (每个时间步的向量维度)
# hidden_size=20 (隐藏状态维度)
# num_layers=2 (堆叠2层)
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)

# batch_first=True: 输入 shape → (batch, seq_len, features)
x = torch.randn(4, 5, 10)    # 4条序列, 每条5步, 每步10维
output, h_n = rnn(x)

print(f"   输入: {x.shape}")
print(f"   输出: {output.shape}   (所有时间步的隐藏状态)")
print(f"   最终隐藏: {h_n.shape}   (2层 × 4个样本 × 20维)")
print()

# ============================================================
# 2. LSTM — 长短期记忆
# ============================================================
print("2. LSTM — 解决长程依赖")
print("=" * 60)

lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, (h_n, c_n) = lstm(x)

print(f"   LSTM 多了细胞状态 c: {c_n.shape}")
print(f"   h 是短期记忆, c 是长期记忆")
print(f"   参数量 (每层): 4×(10×20 + 20×20 + 20) = "
      f"每个门:输入权重+隐藏权重+偏置")
print()

# ============================================================
# 3. GRU — 门控循环单元 (LSTM简化版)
# ============================================================
print("3. GRU — 比LSTM轻量")
print("=" * 60)

gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, h_n = gru(x)

lstm_params = sum(p.numel() for p in lstm.parameters())
gru_params = sum(p.numel() for p in gru.parameters())
print(f"   LSTM 参数: {lstm_params}, GRU 参数: {gru_params}")
print(f"   GRU 省 25% 参数 (3个门 vs 4个门)")
print()

# ============================================================
# 4. 实战: 多肽溶解度预测 (LSTM + 序列)
# ============================================================
print("4. 实战: 多肽溶解度预测")
print("=" * 60)


class PeptideLSTM(nn.Module):
    """
    输入: 氨基酸序列 (每残基用one-hot编码)
    输出: 预测 logS (溶解度)
    架构: Embedding → LSTM → FC → 输出
    """
    def __init__(self, vocab_size=20, embed_dim=32, hidden_dim=64,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=True)      # 双向LSTM
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),           # *2 因为双向
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),                         # 输出单个logS值
        )

    def forward(self, x):
        # x: (batch, seq_len) — 每个元素是AA编号 (0-19)
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)     # (batch, seq_len, hidden*2)

        # 取最后一个时间步的输出 (或用mean pooling)
        last_out = lstm_out[:, -1, :]          # (batch, hidden*2)

        logS = self.fc(last_out)               # (batch, 1)
        return logS.squeeze(-1)                # (batch,)


# 模拟数据
torch.manual_seed(42)
n_train, n_test = 200, 50
seq_len_range = (3, 15)

# 生成随机多肽序列
def random_peptides(n, min_len, max_len):
    lengths = torch.randint(min_len, max_len + 1, (n,))
    max_l = lengths.max()
    seqs = torch.randint(0, 20, (n, max_l))     # 20种AA
    # 用 padding mask 标记有效位置
    mask = torch.arange(max_l).unsqueeze(0) < lengths.unsqueeze(1)
    return seqs, mask, lengths


X_train, mask_train, len_train = random_peptides(n_train, *seq_len_range)
X_test, mask_test, len_test = random_peptides(n_test, *seq_len_range)

# 模拟标签: logS 与序列长度和组成有关
y_train = -3.0 + 0.1 * len_train.float() + torch.randn(n_train) * 0.5
y_test = -3.0 + 0.1 * len_test.float() + torch.randn(n_test) * 0.5

print(f"   训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"   标签范围: [{y_train.min():.1f}, {y_train.max():.1f}]")

# 训练
model = PeptideLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(1, 51):
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)
            mae = (test_pred - y_test).abs().mean()
        print(f"   Epoch {epoch:2d}: train_loss={loss.item():.4f}, "
              f"test_loss={test_loss.item():.4f}, MAE={mae.item():.3f}")

print()

# ============================================================
# 5. RNN vs LSTM vs GRU 速查
# ============================================================
print("5. RNN/LSTM/GRU 对比")
print("=" * 60)
print("""
   RNN:  简单但梯度消失, 只能处理短序列
   LSTM: 3个门(遗忘/输入/输出) + 细胞状态, 处理长程
   GRU:  2个门(重置/更新), 省参数, 效果接近LSTM
   双向:  同时从前向后和从后向前读, 适合分类(不能用于生成)
""")

print("[OK] PyTorch 模块6 完成!")
