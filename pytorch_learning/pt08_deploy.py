"""
PyTorch 模块8：模型保存、加载与部署
=========================================
checkpoint、state_dict、导出ONNX、TorchScript、推理优化。
"""

import torch
import torch.nn as nn
import os
import tempfile

# ============================================================
# 1. 保存和加载模型
# ============================================================
print("1. 模型保存与加载")
print("=" * 60)

# 一个简单模型
model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---- 方式A: 只保存权重 (推荐) ----
save_dir = os.path.join(tempfile.gettempdir(), "pytorch_demo")
os.makedirs(save_dir, exist_ok=True)

weight_path = os.path.join(save_dir, "model_weights.pth")
torch.save(model.state_dict(), weight_path)
print(f"   已保存权重到: {weight_path}")

# 加载: 先创建同结构模型, 再加载权重
new_model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 2),
)
new_model.load_state_dict(torch.load(weight_path, weights_only=True))
print(f"   已加载权重, 参数一致: {all(torch.equal(p1, p2) for p1, p2 in zip(model.parameters(), new_model.parameters()))}")

# ---- 方式B: 保存整个模型 (简单但不推荐跨版本) ----
full_path = os.path.join(save_dir, "model_full.pth")
torch.save(model, full_path)
loaded = torch.load(full_path, weights_only=False)
print(f"   完整模型加载成功: {type(loaded).__name__}")

# ============================================================
# 2. Checkpoint — 断点续训
# ============================================================
print("\n\n2. Checkpoint — 保存训练状态")
print("=" * 60)

checkpoint = {
    'epoch': 50,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.0234,
    'best_val_acc': 0.95,
}
ckpt_path = os.path.join(save_dir, "checkpoint.pth")
torch.save(checkpoint, ckpt_path)
print(f"   已保存checkpoint: epoch=50, loss=0.0234, acc=0.95")

# 恢复训练
ckpt = torch.load(ckpt_path, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
start_epoch = ckpt['epoch'] + 1
print(f"   恢复训练: 从 epoch {start_epoch} 开始")

# ============================================================
# 3. 推理模式
# ============================================================
print("\n\n3. 推理优化")
print("=" * 60)

x = torch.randn(8, 10)

# ---- model.eval() — 关闭 Dropout/BatchNorm ----
model.eval()
with torch.no_grad():                          # 不计算梯度
    y = model(x)
print(f"   eval+no_grad 推理: {x.shape} → {y.shape}")
print(f"   eval(): 冻结BatchNorm的running stats, 固定Dropout")
print(f"   no_grad(): 关闭autograd, 省显存+加速")

# ---- torch.inference_mode() — 更激进 (PyTorch 1.9+) ----
with torch.inference_mode():
    y2 = model(x)
print(f"   inference_mode(): 比no_grad更严格, 完全禁止autograd")
print()

# ============================================================
# 4. 导出 ONNX (跨框架部署)
# ============================================================
print("4. 导出 ONNX")
print("=" * 60)

onnx_path = os.path.join(save_dir, "model.onnx")
dummy_input = torch.randn(1, 10)

try:
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    )
    print(f"   已导出 ONNX: {onnx_path}")
    print(f"   动态batch: 支持任意batch大小")
except Exception as e:
    print(f"   ONNX导出跳过: {e}")

print()

# ============================================================
# 5. TorchScript — 生产部署
# ============================================================
print("5. TorchScript — C++ 运行时")
print("=" * 60)

# trace: 记录一次前向传播的计算图
example = torch.randn(1, 10)
traced = torch.jit.trace(model.eval(), example)
script_path = os.path.join(save_dir, "model_scripted.pt")
traced.save(script_path)
print(f"   TorchScript 已保存: {script_path}")

# 加载, 无需 Python 源码
loaded_script = torch.jit.load(script_path)
out = loaded_script(example)
print(f"   脚本模型推理: {out.shape}")

# ---- 对比 trace vs script ----
print(f"\n   trace:  记录实际操作路径 (快, 但不能有控制流)")
print(f"   script: 解析Python代码转IR (慢, 但支持if/for)")

# ============================================================
# 6. 模型压缩 — 半精度 & 量化
# ============================================================
print("\n\n6. 模型压缩")
print("=" * 60)

# FP16 半精度推理
model_fp16 = model.half()
x_fp16 = torch.randn(8, 10).half()
with torch.no_grad():
    y_fp16 = model_fp16(x_fp16)
print(f"   FP16 推理: {x_fp16.dtype} → {y_fp16.dtype}")
print(f"   好处: 显存减半, 速度提升 (需要GPU支持Tensor Cores)")

# 动态量化 (CPU推理加速)
try:
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    print(f"   动态量化: Linear层权重 int8, 模型缩小~4x")
except:
    print(f"   量化跳过 (部分模型结构不支持)")

# 清理
import shutil
shutil.rmtree(save_dir, ignore_errors=True)

print(f"\n[OK] PyTorch 模块8 完成!")
print(f"\n=== PyTorch 8 个模块全部完成! ===")
