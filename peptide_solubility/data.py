"""
data.py — 数据加载与预处理
支持加载带标签数据用于ML训练，无数据时提供经验规则预测。
"""

import numpy as np
from .features import extract_all_features
from .utils import parse_input, sequence_to_smiles


def load_training_data(csv_path):
    """
    从 CSV 加载训练数据。
    期望列: 'sequence' 或 'smiles', 以及 'logS' (溶解度标签)。

    返回: (X: np.array 特征矩阵, y: np.array 标签, names: [str] 输入列表)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    # 识别输入列 (sequence 或 smiles)
    input_col = None
    input_type = None
    if 'sequence' in df.columns:
        input_col = 'sequence'
        input_type = 'sequence'
    elif 'smiles' in df.columns:
        input_col = 'smiles'
        input_type = 'smiles'
    elif 'seq' in df.columns:
        input_col = 'seq'
        input_type = 'sequence'
    else:
        # 尝试用第一列
        input_col = df.columns[0]
        sample = str(df.iloc[0, 0])
        tp, _ = parse_input(sample)
        input_type = tp if tp else 'sequence'

    # 识别标签列
    label_col = None
    for col in ['logS', 'solubility', 'LogS', 'measured log solubility', 'label']:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        # 尝试最后一列
        label_col = df.columns[-1]

    inputs = df[input_col].astype(str).tolist()
    y = df[label_col].astype(float).values

    # 提取特征
    X_list = []
    valid_indices = []
    for i, inp in enumerate(inputs):
        try:
            feats, _ = extract_all_features(inp, input_type)
            X_list.append(feats)
            valid_indices.append(i)
        except Exception:
            continue

    X = np.array(X_list)
    y = y[valid_indices]

    return X, y, [inputs[i] for i in valid_indices]


# ================================================================
#  经验规则预测 (兜底方案)
# ================================================================

def rule_based_logS(sequence_or_smiles, input_type=None):
    """
    基于经验规则估算多肽 logS。
    规则: logS ≈ a*avg_hydrophobicity + b*1/nresidues + c*net_charge + d

    参考:
    - Eisenberg 疏水性标度与溶解度负相关
    - 短肽 > 长肽 (一般)
    - 净电荷增加溶解度

    返回: (logS_mol_L, category)
      category: 'high'(> -2), 'medium'(-2~-4), 'low'(< -4), 单位 log(mol/L)
    """
    from .features import _peptide_features
    from .utils import parse_input

    if input_type is None:
        input_type, canonical = parse_input(sequence_or_smiles)
    else:
        canonical = sequence_or_smiles

    if canonical is None:
        raise ValueError(f"无法解析: {sequence_or_smiles}")

    # 获取肽特征
    feats, names = _peptide_features(canonical, input_type)

    # 解包关键特征
    n_residues = feats[0]      # 长度
    gravy = feats[4]           # GRAVY
    charge = feats[2]          # 净电荷

    # 经验公式 (系数来自文献近似)
    # logS ≈ -0.5 * GRAVY - 1.0 * log10(n_residues) - 0.3 * abs(charge) - 2.0
    if n_residues <= 1:
        logS = -1.0  # 单个氨基酸(无肽键)
    else:
        logS = -0.5 * gravy - 1.0 * np.log10(max(n_residues, 1)) - \
               0.3 * abs(min(charge, 2)) - 2.0

    # 分类
    if logS > -2.0:
        category = 'high'
    elif logS > -4.0:
        category = 'medium'
    else:
        category = 'low'

    return logS, category, dict(zip(names, feats))
