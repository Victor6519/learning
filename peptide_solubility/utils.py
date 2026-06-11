"""
utils.py — 氨基酸属性表、序列↔SMILES 转换、输入解析
"""

import re
from rdkit import Chem

# ================================================================
# 1. 20 种标准氨基酸属性表
# ================================================================
# 结构: { 单字母: {三字母, 全名, 分子量, 侧链pKa, 疏水性(Kyte-Doolittle), 侧链类型, 不稳定指数权重} }

AMINO_ACIDS = {
    'A': {'name3': 'ALA', 'full': 'Alanine',      'mw': 89.09,  'pKa': None, 'hydropathy':  1.8, 'type': 'aliphatic', 'instability': 0.0},
    'R': {'name3': 'ARG', 'full': 'Arginine',      'mw': 174.20, 'pKa': 12.5,  'hydropathy': -4.5, 'type': 'basic',     'instability': 0.0},
    'N': {'name3': 'ASN', 'full': 'Asparagine',    'mw': 132.12, 'pKa': None,  'hydropathy': -3.5, 'type': 'polar',     'instability': 0.0},
    'D': {'name3': 'ASP', 'full': 'Aspartic acid', 'mw': 133.10, 'pKa': 3.9,   'hydropathy': -3.5, 'type': 'acidic',    'instability': 0.0},
    'C': {'name3': 'CYS', 'full': 'Cysteine',      'mw': 121.16, 'pKa': 8.3,   'hydropathy':  2.5, 'type': 'special',   'instability': 0.0},
    'Q': {'name3': 'GLN', 'full': 'Glutamine',     'mw': 146.15, 'pKa': None,  'hydropathy': -3.5, 'type': 'polar',     'instability': 0.0},
    'E': {'name3': 'GLU', 'full': 'Glutamic acid', 'mw': 147.13, 'pKa': 4.3,   'hydropathy': -3.5, 'type': 'acidic',    'instability': 0.0},
    'G': {'name3': 'GLY', 'full': 'Glycine',       'mw': 75.07,  'pKa': None,  'hydropathy': -0.4, 'type': 'special',   'instability': 0.0},
    'H': {'name3': 'HIS', 'full': 'Histidine',     'mw': 155.16, 'pKa': 6.0,   'hydropathy': -3.2, 'type': 'basic',     'instability': 0.0},
    'I': {'name3': 'ILE', 'full': 'Isoleucine',    'mw': 131.17, 'pKa': None,  'hydropathy':  4.5, 'type': 'aliphatic', 'instability': 0.0},
    'L': {'name3': 'LEU', 'full': 'Leucine',       'mw': 131.17, 'pKa': None,  'hydropathy':  3.8, 'type': 'aliphatic', 'instability': 0.0},
    'K': {'name3': 'LYS', 'full': 'Lysine',        'mw': 146.19, 'pKa': 10.5,  'hydropathy': -3.9, 'type': 'basic',     'instability': 0.0},
    'M': {'name3': 'MET', 'full': 'Methionine',    'mw': 149.21, 'pKa': None,  'hydropathy':  1.9, 'type': 'aliphatic', 'instability': 0.0},
    'F': {'name3': 'PHE', 'full': 'Phenylalanine', 'mw': 165.19, 'pKa': None,  'hydropathy':  2.8, 'type': 'aromatic',  'instability': 0.0},
    'P': {'name3': 'PRO', 'full': 'Proline',       'mw': 115.13, 'pKa': None,  'hydropathy': -1.6, 'type': 'special',   'instability': 0.0},
    'S': {'name3': 'SER', 'full': 'Serine',        'mw': 105.09, 'pKa': None,  'hydropathy': -0.8, 'type': 'polar',     'instability': 0.0},
    'T': {'name3': 'THR', 'full': 'Threonine',     'mw': 119.12, 'pKa': None,  'hydropathy': -0.7, 'type': 'polar',     'instability': 0.0},
    'W': {'name3': 'TRP', 'full': 'Tryptophan',    'mw': 204.23, 'pKa': None,  'hydropathy': -0.9, 'type': 'aromatic',  'instability': 0.0},
    'Y': {'name3': 'TYR', 'full': 'Tyrosine',      'mw': 181.19, 'pKa': 10.1,  'hydropathy': -1.3, 'type': 'aromatic',  'instability': 0.0},
    'V': {'name3': 'VAL', 'full': 'Valine',        'mw': 117.15, 'pKa': None,  'hydropathy':  4.2, 'type': 'aliphatic', 'instability': 0.0},
}

# 不稳定指数权重 (Guruprasad 1990)
INSTABILITY_WEIGHTS = {
    'D': 1.0, 'P': 0.5, 'E': 1.0, 'A': 0.0, 'C': 0.0,
    'M': 0.0, 'T': 1.0, 'V': 0.0, 'S': 1.0, 'W': 0.0,
    'Y': 1.0, 'F': 1.0, 'L': 0.0, 'N': 1.0, 'G': 1.0,
    'R': 1.0, 'H': 2.0, 'I': 0.0, 'K': 1.0, 'Q': 1.0,
}

# 三字母→单字母映射
NAME3_TO_1 = {v['name3']: k for k, v in AMINO_ACIDS.items()}

# ================================================================
# 2. 序列 ↔ SMILES 转换
# ================================================================

def sequence_to_smiles(seq):
    """
    氨基酸序列 → SMILES。
    支持格式: 'ALA-GLY-LYS' 或 'AGK' 或 'AlaGlyLys'
    """
    clean = _normalize_sequence(seq)
    if not clean:
        return None
    mol = Chem.MolFromSequence(clean)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def smiles_to_sequence(smi):
    """
    从 SMILES 反推氨基酸序列。
    注意: RDKit 仅在 MolFromSequence/MolFromFASTA 创建的分子上保留残基信息。
    对规范化 SMILES 通常返回 None, 因此 SMILES 输入时以 graph-based 特征替代。
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # 尝试 RDKit 内置方法
    try:
        seq = Chem.MolToSequence(mol)
        if seq:
            return seq
    except:
        pass
    try:
        fasta = Chem.MolToFASTA(mol)
        if fasta:
            return fasta
    except:
        pass
    return None


def count_amide_bonds(mol):
    """从 SMILES 的分子图中统计酰胺键 NC(=O) 数量 (≈ 肽键数)"""
    amide = Chem.MolFromSmarts('NC(=O)')
    if mol is None or amide is None:
        return 0
    return len(mol.GetSubstructMatches(amide))


def _normalize_sequence(text):
    """
    将各种序列格式统一为单字母大写。
    'ALA-GLY-LYS' → 'AGK'
    'AlaGlyLys'   → 'AGK'
    'AGK'         → 'AGK'
    """
    text = text.strip().upper()
    # 移除分隔符和空格
    if '-' in text or ' ' in text:
        parts = re.split(r'[- ]+', text)
        result = []
        for p in parts:
            if len(p) == 3 and p in NAME3_TO_1:
                result.append(NAME3_TO_1[p])
            elif len(p) == 1 and p in AMINO_ACIDS:
                result.append(p)
            else:
                result.append(p)  # 保持原样
        return ''.join(result)
    # 纯单字母序列
    if all(c in AMINO_ACIDS for c in text):
        return text
    # 混合格式 (AlaGlyLys), 每个残基首字母大写
    # 简单处理: 转大写后逐字检查
    if all(c in AMINO_ACIDS for c in text.upper()):
        return text.upper()
    return text



# ================================================================
# 3. 输入解析 — 自动识别 SMILES vs 序列
# ================================================================

def parse_input(text):
    """
    自动识别输入类型并返回 (type, canonical_text)。
    type: 'smiles' | 'sequence'
    canonical_text: SMILES字符串 或 单字母序列
    """
    text = text.strip()
    if not text:
        return None, None

    # 1. 尝试作为序列解析
    clean_seq = _normalize_sequence(text)
    if clean_seq and all(c in AMINO_ACIDS for c in clean_seq) and len(clean_seq) >= 1:
        return 'sequence', clean_seq

    # 2. 尝试作为 SMILES 解析
    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        # 检查是否像肽 (存在多个酰胺键 NC=O)
        amide_pat = Chem.MolFromSmarts('NC(=O)')
        if mol.HasSubstructMatch(amide_pat):
            return 'smiles', Chem.MolToSmiles(mol, canonical=True)

    # 3. 再尝试一次: 作为SMILES可能有多个肽键
    if mol is not None:
        return 'smiles', Chem.MolToSmiles(mol, canonical=True)

    # 4. 都失败, 当序列处理
    if len(clean_seq) >= 1 and all(c in AMINO_ACIDS for c in clean_seq):
        return 'sequence', clean_seq

    return None, None


def combine_fragment_peptide(smi_fragment, aa_sequence):
    """
    将小分子 SMILES 片段和多肽序列拼接成完整缀合物 SMILES。
    小分子的羧基和多肽 N 端形成酰胺键。
    返回: 完整 SMILES 或 None (拼接失败时)
    """
    mol_frag = Chem.MolFromSmiles(smi_fragment)
    mol_pep = Chem.MolFromSequence(aa_sequence)

    if mol_frag is None or mol_pep is None:
        return None

    result = _connect_amide(mol_frag, mol_pep)
    if result:
        return Chem.MolToSmiles(result)
    return None


# ================================================================
# 4. [] 标记解析与多段拼接
#    支持: AGK[SMILES]LFP[SMILES]RR 格式
# ================================================================

def parse_bracketed_input(text):
    """
    解析带 {} 标记的混合输入 (用{}避免和SMILES的[]冲突)。
    'AGK{N[C@@H](C(=O)O)c1ccccc1}LFP' → [seq, smi, seq]
    纯序列无 {} 也正常返回单段。
    """
    import re
    text = text.strip()
    if not text:
        return []

    # 正则: 匹配 {...} 内的 SMILES 片段
    pattern = re.compile(r'\{([^}]+)\}')
    segments = []

    pos = 0
    for m in pattern.finditer(text):
        # [] 之前的部分 → 氨基酸序列
        before = text[pos:m.start()]
        if before.strip():
            # 转为纯单字母
            clean = _normalize_sequence(before)
            segments.append({'type': 'seq', 'val': clean})

        # [] 内的部分 → SMILES 片段
        smi = m.group(1).strip()
        mol = Chem.MolFromSmiles(smi)
        if mol:
            segments.append({'type': 'smi', 'val': Chem.MolToSmiles(mol, canonical=True)})
        else:
            segments.append({'type': 'smi', 'val': smi})  # 保持原样, 后面会报错

        pos = m.end()

    # 最后一段 → 氨基酸序列
    tail = text[pos:].strip()
    if tail:
        clean = _normalize_sequence(tail)
        segments.append({'type': 'seq', 'val': clean})

    if not segments:
        # 无 [], 整个当纯序列
        clean = _normalize_sequence(text)
        if clean:
            segments.append({'type': 'seq', 'val': clean})

    return segments


def _mol_from_segment(seg):
    """将段转为 Mol 对象"""
    if seg['type'] == 'seq':
        return Chem.MolFromSequence(seg['val'])
    else:
        return Chem.MolFromSmiles(seg['val'])


def _connect_amide(mol_a, mol_b):
    """
    用酰胺键连接两个分子。自动检测方向:
    - 先试 mol_a-COOH + mol_b-NH2
    - 再试 mol_a-NH2 + mol_b-COOH
    返回合并后的 Mol 或 None。
    """
    from rdkit.Chem import AllChem

    rxn = AllChem.ReactionFromSmarts(
        '[C:1](=[O:2])[OH:3].[N:4][H:5]>>[C:1](=[O:2])[N:4].O'
    )

    # 方向1: A的COOH + B的NH2
    products = rxn.RunReactants((Chem.AddHs(mol_a), Chem.AddHs(mol_b)))
    if products:
        for mol in products[0]:
            try:
                Chem.SanitizeMol(mol)
            except:
                pass
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            if smi != 'O' and Chem.MolFromSmiles(smi):
                return Chem.RemoveHs(mol)

    # 方向2: A的NH2 + B的COOH
    products = rxn.RunReactants((Chem.AddHs(mol_b), Chem.AddHs(mol_a)))
    if products:
        for mol in products[0]:
            try:
                Chem.SanitizeMol(mol)
            except:
                pass
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            if smi != 'O' and Chem.MolFromSmiles(smi):
                return Chem.RemoveHs(mol)

    return None


def assemble_molecule(segments):
    """
    将段列表逐段拼接成完整分子。
    返回: 完整 SMILES 或 None。
    """
    if not segments:
        return None

    if len(segments) == 1:
        seg = segments[0]
        mol = _mol_from_segment(seg)
        if mol:
            return Chem.MolToSmiles(mol)
        return seg['val']  # fallback

    # 多段: 逐段酰胺键连接
    current = _mol_from_segment(segments[0])
    if current is None:
        return None

    for seg in segments[1:]:
        next_mol = _mol_from_segment(seg)
        if next_mol is None:
            return None
        combined = _connect_amide(current, next_mol)
        if combined is None:
            # 最后一招: 直接拼 SMILES 字符串用 '.' 连起来
            smi_a = Chem.MolToSmiles(current)
            smi_b = Chem.MolToSmiles(next_mol)
            combined = Chem.MolFromSmiles(f'{smi_a}.{smi_b}')
            if combined is None:
                return None
        current = combined

    try:
        Chem.SanitizeMol(current)
    except:
        pass
    return Chem.MolToSmiles(current)
