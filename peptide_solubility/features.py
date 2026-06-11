"""
features.py — 特征提取: RDKit描述符 + 多肽专属特征
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

from .utils import AMINO_ACIDS, INSTABILITY_WEIGHTS


def extract_all_features(input_text, input_type=None):
    """
    统一特征提取入口。
    input_type: 'smiles' | 'sequence' | None (自动)

    返回: feature_vector (np.array), feature_names (list)
    """
    from .utils import parse_input

    if input_type is None:
        input_type, canonical = parse_input(input_text)
    else:
        canonical = input_text

    if canonical is None:
        raise ValueError(f"无法解析输入: {input_text}")

    rdkit_feats, rdkit_names = _rdkit_features(canonical, input_type)
    peptide_feats, peptide_names = _peptide_features(canonical, input_type)

    all_feats = np.concatenate([rdkit_feats, peptide_feats])
    all_names = rdkit_names + peptide_names

    return all_feats, all_names


# ====================================================================
#  RDKit 描述符特征
# ====================================================================

def _rdkit_features(canonical, input_type):
    """在 SMILES 上计算 RDKit 描述符"""

    if input_type == 'sequence':
        from .utils import sequence_to_smiles
        smi = sequence_to_smiles(canonical)
        mol = Chem.MolFromSmiles(smi) if smi else None
    else:
        mol = Chem.MolFromSmiles(canonical)

    if mol is None:
        return np.zeros(8), ['rdkit_failed'] * 8

    feats = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
    ])

    names = [
        'MolWt', 'MolLogP', 'TPSA', 'NumHBD', 'NumHBA',
        'NumRotBonds', 'FracCSP3', 'HeavyAtomCount'
    ]

    return feats, names


# ====================================================================
#  多肽专属特征
# ====================================================================

def _peptide_features(canonical, input_type):
    """在氨基酸序列上计算肽专属特征"""

    if input_type == 'smiles':
        from .utils import smiles_to_sequence
        seq = smiles_to_sequence(canonical)
        if seq is None:
            return _smiles_peptide_features(canonical)
    else:
        seq = canonical

    if not seq or len(seq) == 0:
        return np.zeros(15), ['empty_seq'] * 15

    length = len(seq)

    # 获取每种氨基酸的信息
    residues = [AMINO_ACIDS.get(aa, {}) for aa in seq]
    valid = [r for r in residues if r]

    if not valid:
        return np.zeros(15), ['invalid_seq'] * 15

    # 1. 肽链长度
    n_residues = length

    # 2. 序列分子量 (MW = sum(aa_MW) - 18*(n-1) 脱去水)
    sum_mw = sum(r.get('mw', 0) for r in residues)
    seq_mw = sum_mw - 18.015 * (length - 1)

    # 3. 净电荷 @ pH 7
    charge = _calc_net_charge(seq, residues)

    # 4. 平均疏水性 (Kyte-Doolittle)
    hydro_values = [r.get('hydropathy', 0) for r in residues]
    avg_hydro = np.mean(hydro_values)

    # 5. GRAVY 指数
    gravy = sum(hydro_values) / length

    # 6. 等电点 pI (迭代法)
    pi = _calc_pi(seq, residues)

    # 7. 不稳定指数
    instability = _calc_instability(seq)

    # 8. 脂肪指数
    aliphatic = _calc_aliphatic_index(seq, residues)

    # 9. 芳香残基比例
    aromatic_count = sum(1 for r in residues if r.get('type') == 'aromatic')
    aromatic_ratio = aromatic_count / length

    # 10. 酸性残基比例
    acidic_count = sum(1 for r in residues if r.get('type') == 'acidic')
    acidic_ratio = acidic_count / length

    # 11. 碱性残基比例
    basic_count = sum(1 for r in residues if r.get('type') == 'basic')
    basic_ratio = basic_count / length

    # 12. 极性残基比例 (polar + acidic + basic)
    polar_count = sum(1 for r in residues
                     if r.get('type') in ('polar', 'acidic', 'basic'))
    polar_ratio = polar_count / length

    # 13. N端残基疏水性
    n_term_hydro = residues[0].get('hydropathy', 0) if residues else 0

    # 14. C端残基疏水性
    c_term_hydro = residues[-1].get('hydropathy', 0) if residues else 0

    # 15. 电荷不对称性: (正电荷 - 负电荷) / 总电荷
    pos_charged = sum(1 for r in residues if r.get('type') == 'basic')
    neg_charged = sum(1 for r in residues if r.get('type') == 'acidic')
    charge_asym = (pos_charged - neg_charged) / max(abs(charge), 1)

    feats = np.array([
        n_residues, seq_mw, charge, avg_hydro, gravy, pi,
        instability, aliphatic, aromatic_ratio, acidic_ratio,
        basic_ratio, polar_ratio, n_term_hydro, c_term_hydro, charge_asym
    ])

    names = [
        'N_Residues', 'Seq_MW', 'Net_Charge_pH7', 'Avg_Hydrophobicity',
        'GRAVY', 'pI', 'Instability_Index', 'Aliphatic_Index',
        'Aromatic_Ratio', 'Acidic_Ratio', 'Basic_Ratio', 'Polar_Ratio',
        'N_Term_Hydro', 'C_Term_Hydro', 'Charge_Asymmetry'
    ]

    return feats, names


# ====================================================================
#  子函数: 电荷、pI、不稳定指数、脂肪指数
# ====================================================================

def _calc_net_charge(seq, residues):
    """pH 7 净电荷 (简化 Henderson-Hasselbalch)"""
    charge = 0.0

    # N 端 (+1 @ pH 7, pKa ~9.7)
    if len(seq) > 0:
        charge += 1.0 / (1 + 10 ** (7.0 - 9.7))

    # C 端 (-1 @ pH 7, pKa ~2.2)
    if len(seq) > 0:
        charge -= 1.0 / (1 + 10 ** (2.2 - 7.0))

    # 侧链
    for r in residues:
        pKa = r.get('pKa')
        if pKa is None:
            continue
        aa_type = r.get('type')
        if aa_type == 'acidic':   # Asp, Glu: 负电荷
            charge -= 1.0 / (1 + 10 ** (pKa - 7.0))
        elif aa_type == 'basic':  # His, Lys, Arg: 正电荷
            if r.get('name3') == 'HIS':
                charge += 1.0 / (1 + 10 ** (7.0 - pKa))
            else:
                charge += 1.0 / (1 + 10 ** (7.0 - pKa))

    return charge


def _calc_pi(seq, residues):
    """迭代法找等电点 (pI)"""
    # 简化版: 基于带电残基数目估计
    # pI ≈ (pKa_acidic_avg + pKa_basic_avg) / 2, 修正到合理范围

    acidic_pKas = []
    basic_pKas_pairs = []

    # N端
    acidic_pKas.append(9.7)  # 简化为 N-terminal
    # C端
    basic_pKas_pairs.append(2.2)  # 简化为 C-terminal

    for r in residues:
        pKa = r.get('pKa')
        if pKa is None:
            continue
        if r.get('type') == 'acidic':
            acidic_pKas.append(pKa)
        elif r.get('type') == 'basic':
            basic_pKas_pairs.append(pKa)

    # 对有电荷的残基做二分搜索找 pI
    all_pkas = sorted(acidic_pKas + basic_pKas_pairs)
    if not all_pkas:
        return 7.0

    # 在候选 pKa 中找净电荷过零点的
    lo, hi = min(all_pkas) - 0.5, max(all_pkas) + 0.5
    for _ in range(50):
        mid = (lo + hi) / 2
        q = _charge_at_pH(seq, residues, mid)
        if q > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _charge_at_pH(seq, residues, pH):
    """给定 pH 计算净电荷"""
    q = 0.0
    if len(seq) > 0:
        q += 1.0 / (1 + 10 ** (pH - 9.7))       # N端
        q -= 1.0 / (1 + 10 ** (2.2 - pH))        # C端
    for r in residues:
        pKa = r.get('pKa')
        if pKa is None:
            continue
        if r.get('type') == 'acidic':
            q -= 1.0 / (1 + 10 ** (pKa - pH))
        elif r.get('type') == 'basic':
            if r.get('name3') == 'HIS':
                q += 1.0 / (1 + 10 ** (pH - pKa))
            else:
                q += 1.0 / (1 + 10 ** (pH - pKa))
    return q


def _calc_instability(seq):
    """Guruprasad 1990 不稳定指数"""
    if len(seq) < 2:
        return 0.0
    total = 0.0
    for i in range(len(seq) - 1):
        dipeptide = seq[i:i+2]
        w1 = INSTABILITY_WEIGHTS.get(dipeptide[0], 0)
        w2 = INSTABILITY_WEIGHTS.get(dipeptide[1], 0)
        total += w1 + w2
    return (10.0 / len(seq)) * total


def _calc_aliphatic_index(seq, residues):
    """Ikai 1980 脂肪指数"""
    ala = sum(1 for r in residues if r.get('name3') == 'ALA')
    val = sum(1 for r in residues if r.get('name3') == 'VAL')
    ile = sum(1 for r in residues if r.get('name3') == 'ILE')
    leu = sum(1 for r in residues if r.get('name3') == 'LEU')
    n = len(seq)
    if n == 0:
        return 0.0
    return (ala + 2.4 * val + 3.9 * ile + 3.9 * leu) * 100.0 / max(n, 1)


# ====================================================================
#  SMILES 分子图 → 肽特征 (无需知道具体残基)
# ====================================================================

def _smiles_peptide_features(smi):
    """
    从 SMILES 的分子图直接提取肽相关特征。
    不需要知道具体氨基酸, 靠统计子结构和拓扑性质。
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(15), ['mol_failed'] * 15

    # 1. 酰胺键 NC(=O) 数量 ≈ 肽键数
    amide = Chem.MolFromSmarts('NC(=O)')
    n_peptide_bonds = len(mol.GetSubstructMatches(amide)) if amide else 0

    # 2. 估计残基数 ≈ 肽键数 + 1
    n_residues = max(n_peptide_bonds + 1, 1)

    # 3. 序列分子量 (直接用 RDKit 算的 MolWt)
    seq_mw = Descriptors.MolWt(mol)

    # 4. 净电荷 pH7 — 统计可电离基团
    # 碱性: 氨基 NH2 (非酰胺的), 胍基 NC(=N)N
    # 酸性: 羧基 C(=O)OH, 非酰胺连接的
    nh2_pat = Chem.MolFromSmarts('[NH2]')
    carboxylic_pat = Chem.MolFromSmarts('C(=O)[OH]')
    n_nh2 = len(mol.GetSubstructMatches(nh2_pat)) if nh2_pat else 0
    n_cooh = len(mol.GetSubstructMatches(carboxylic_pat)) if carboxylic_pat else 0
    # 简化: N端 +1, C端 -1, 侧链 +NH2 -COOH
    charge = n_nh2 * 0.9 - n_cooh * 0.9

    # 5. 平均疏水性 — 从 MolLogP 和 TPSA 反推
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    # logP 越高疏水性越强, TPSA 越高亲水性越强
    avg_hydro = 0.5 * logp - 0.01 * tpsa

    # 6. GRAVY: 平均疏水度 ≈ 总 LogP / MW 比例
    gravy = avg_hydro

    # 7. pI — 基于净电荷估计
    pi = 7.0 - 0.5 * charge  # 带电基团多→ pI偏离7

    # 8. 不稳定指数 — 基于可旋转键数近似
    n_rot = Descriptors.NumRotatableBonds(mol)
    instability = n_rot * 2.0

    # 9. 脂肪指数 — FractionCSP3 代理
    frac_csp3 = Descriptors.FractionCSP3(mol)
    aliphatic = frac_csp3 * 100

    # 10-12. 芳香/酸性/碱性比例
    aromatic_pat = Chem.MolFromSmarts('c')  # 芳香碳
    n_aromatic = len(mol.GetSubstructMatches(aromatic_pat)) if aromatic_pat else 0
    aromatic_ratio = n_aromatic / max(mol.GetNumHeavyAtoms(), 1)
    acidic_ratio = n_cooh / max(n_residues, 1)
    basic_ratio = n_nh2 / max(n_residues, 1)
    polar_ratio = tpsa / max(mol.GetNumHeavyAtoms() * 20, 1)

    # 13-14. N端/C端疏水性 — 用 logP 代理
    n_term_hydro = logp / max(n_residues, 1)
    c_term_hydro = logp / max(n_residues, 1)

    # 15. 电荷不对称性
    charge_asym = (n_nh2 - n_cooh) / max(abs(charge), 1) if abs(charge) > 0.001 else 0.0

    feats = np.array([
        n_residues, seq_mw, charge, avg_hydro, gravy, pi,
        instability, aliphatic, aromatic_ratio, acidic_ratio,
        basic_ratio, polar_ratio, n_term_hydro, c_term_hydro, charge_asym
    ])

    names = [
        'N_Residues', 'Seq_MW', 'Net_Charge_pH7', 'Avg_Hydrophobicity',
        'GRAVY', 'pI', 'Instability_Index', 'Aliphatic_Index',
        'Aromatic_Ratio', 'Acidic_Ratio', 'Basic_Ratio', 'Polar_Ratio',
        'N_Term_Hydro', 'C_Term_Hydro', 'Charge_Asymmetry'
    ]

    return feats, names
