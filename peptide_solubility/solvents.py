"""
solvents.py — 溶剂数据库 + Hansen 溶解度参数 (HSP) 计算
=============================================================
支持 8 种常用溶剂: 水, DMF, DCM, NMP, DMSO, 甲醇, 乙醇, 乙腈
"""

import numpy as np

# ================================================================
# 1. 溶剂数据库
# ================================================================
# HSP 单位: MPa^(1/2)
# δd = 色散力, δp = 极性力, δh = 氢键力
# ε  = 介电常数 (相对)
# 数据来源: Hansen Solubility Parameters Handbook + CRC

SOLVENTS = {
    'water': {
        'name': '水 (Water)',
        'd_d': 15.5, 'd_p': 16.0, 'd_h': 42.3,
        'dielectric': 78.4,
        'class': 'protic',      # 质子性溶剂
        'bp': 100.0,            # 沸点 C
        'viscosity': 0.89,      # cP @ 25C
    },
    'dmf': {
        'name': 'N,N-二甲基甲酰胺 (DMF)',
        'd_d': 17.4, 'd_p': 13.7, 'd_h': 11.3,
        'dielectric': 36.7,
        'class': 'aprotic',     # 非质子性
        'bp': 153.0,
        'viscosity': 0.80,
    },
    'dcm': {
        'name': '二氯甲烷 (DCM)',
        'd_d': 18.2, 'd_p': 6.3, 'd_h': 6.1,
        'dielectric': 9.1,
        'class': 'aprotic',
        'bp': 39.6,
        'viscosity': 0.43,
    },
    'nmp': {
        'name': 'N-甲基吡咯烷酮 (NMP)',
        'd_d': 18.0, 'd_p': 12.3, 'd_h': 7.2,
        'dielectric': 32.2,
        'class': 'aprotic',
        'bp': 202.0,
        'viscosity': 1.67,
    },
    'dmso': {
        'name': '二甲基亚砜 (DMSO)',
        'd_d': 18.4, 'd_p': 16.4, 'd_h': 10.2,
        'dielectric': 46.7,
        'class': 'aprotic',
        'bp': 189.0,
        'viscosity': 1.99,
    },
    'methanol': {
        'name': '甲醇 (Methanol)',
        'd_d': 15.1, 'd_p': 12.3, 'd_h': 22.3,
        'dielectric': 32.7,
        'class': 'protic',
        'bp': 64.7,
        'viscosity': 0.54,
    },
    'ethanol': {
        'name': '乙醇 (Ethanol)',
        'd_d': 15.8, 'd_p': 8.8, 'd_h': 19.4,
        'dielectric': 24.5,
        'class': 'protic',
        'bp': 78.4,
        'viscosity': 1.07,
    },
    'acetonitrile': {
        'name': '乙腈 (Acetonitrile)',
        'd_d': 15.3, 'd_p': 18.0, 'd_h': 6.1,
        'dielectric': 37.5,
        'class': 'aprotic',
        'bp': 81.6,
        'viscosity': 0.37,
    },
}

# 简称映射
ALIASES = {
    'h2o': 'water', 'h20': 'water', 'ho2': 'water',
    'meoh': 'methanol', 'etoh': 'ethanol',
    'mecn': 'acetonitrile', 'acn': 'acetonitrile',
    'dichloromethane': 'dcm', 'dimethylformamide': 'dmf',
}

SOLVENT_LIST = list(SOLVENTS.keys())


def get_solvent(name):
    """通过名称/简称获取溶剂信息"""
    key = name.lower().strip()
    key = ALIASES.get(key, key)
    if key in SOLVENTS:
        return key, SOLVENTS[key]
    return None, None


# ================================================================
# 2. 氨基酸残基对 HSP 的贡献
# ================================================================
# 基于文献的残基分组贡献 (单位: MPa^(1/2) 增量, 相对于甘氨酸基线)
# 参考: Breil et al., Hansen Solubility Parameters (4th ed.), 及肽色谱保留数据

# 基线 (每残基): δd ~ 17, δp ~ 8, δh ~ 10
# 各类残基的增量
RESIDUE_HSP_DELTA = {
    # 非极性/脂肪族: δd +, δp 0, δh 0
    'A': {'d_d':  0.5, 'd_p': 0.0, 'd_h': 0.0},    # Ala 甲基
    'V': {'d_d':  1.5, 'd_p': 0.0, 'd_h': 0.0},    # Val 异丙基
    'L': {'d_d':  1.5, 'd_p': 0.0, 'd_h': 0.0},    # Leu
    'I': {'d_d':  1.5, 'd_p': 0.0, 'd_h': 0.0},    # Ile
    'M': {'d_d':  1.0, 'd_p': 0.0, 'd_h': 0.0},    # Met
    'P': {'d_d':  0.5, 'd_p': 0.0, 'd_h': 0.0},    # Pro

    # 芳香族: δd +, δp +, δh +
    'F': {'d_d':  2.0, 'd_p': 2.0, 'd_h': 1.0},
    'W': {'d_d':  2.5, 'd_p': 2.5, 'd_h': 2.0},
    'Y': {'d_d':  2.0, 'd_p': 2.0, 'd_h': 3.0},

    # 极性非电离: δp +++, δh ++
    'N': {'d_d': -0.5, 'd_p': 3.0, 'd_h': 5.0},
    'Q': {'d_d': -0.5, 'd_p': 3.0, 'd_h': 5.0},
    'S': {'d_d': -0.5, 'd_p': 2.0, 'd_h': 4.0},
    'T': {'d_d': -0.5, 'd_p': 2.0, 'd_h': 4.0},
    'G': {'d_d':  0.0, 'd_p': 0.0, 'd_h': 0.0},

    # 酸性: δp ++, δh +++
    'D': {'d_d': -1.0, 'd_p': 5.0, 'd_h': 8.0},
    'E': {'d_d': -1.0, 'd_p': 5.0, 'd_h': 8.0},

    # 碱性: δp +++, δh +++
    'H': {'d_d': -0.5, 'd_p': 4.0, 'd_h': 6.0},
    'K': {'d_d':  0.0, 'd_p': 3.0, 'd_h': 7.0},
    'R': {'d_d': -0.5, 'd_p': 5.0, 'd_h': 8.0},

    # 半胱氨酸: δh 低 (硫醇)
    'C': {'d_d':  1.0, 'd_p': 1.0, 'd_h': 2.0},
}

# 主链基线 (每个残基的主链贡献)
BACKBONE_HSP = {'d_d': 17.0, 'd_p': 8.0, 'd_h': 10.0}


def estimate_peptide_hsp(sequence):
    """
    从氨基酸序列估算肽的 Hansen 溶解度参数。

    返回: {'d_d': float, 'd_p': float, 'd_h': float}
    """
    from .utils import AMINO_ACIDS

    seq = sequence.upper().strip()
    if not seq:
        return BACKBONE_HSP.copy()

    n = len(seq)
    if n == 0:
        return BACKBONE_HSP.copy()

    # 主链贡献加权
    bb = BACKBONE_HSP.copy()

    # 收集侧链增量
    deltas = []
    for aa in seq:
        d = RESIDUE_HSP_DELTA.get(aa, {'d_d': 0, 'd_p': 0, 'd_h': 0})
        deltas.append(d)

    # 侧链平均增量
    avg_d_d = sum(d['d_d'] for d in deltas) / n
    avg_d_p = sum(d['d_p'] for d in deltas) / n
    avg_d_h = sum(d['d_h'] for d in deltas) / n

    # 肽的 HSP = 主链基线 + 侧链平均增量
    hsp = {
        'd_d': bb['d_d'] + avg_d_d,
        'd_p': bb['d_p'] + avg_d_p,
        'd_h': bb['d_h'] + avg_d_h,
    }
    return hsp


# ================================================================
# 3. HSP 距离与溶解度预测
# ================================================================

def hsp_distance(peptide_hsp, solvent_hsp):
    """
    Hansen 溶解度参数距离 (Ra, 单位 MPa^(1/2))。

    Ra² = 4(Δδd)² + (Δδp)² + (Δδh)²

    Ra 越小 → 溶剂-溶质越兼容 → 溶解度越好
    """
    dd = peptide_hsp['d_d'] - solvent_hsp['d_d']
    dp = peptide_hsp['d_p'] - solvent_hsp['d_p']
    dh = peptide_hsp['d_h'] - solvent_hsp['d_h']
    return np.sqrt(4 * dd**2 + dp**2 + dh**2)


def relative_solubility_score(hsp_distance):
    """
    将 HSP 距离映射到溶解性评分 (0~1)。

    Ra < 5: excellent (score ~ 0.9)
    Ra 5-10: good (score ~ 0.6-0.8)
    Ra 10-15: moderate (score ~ 0.3-0.6)
    Ra > 15: poor (score < 0.3)
    """
    # 指数衰减映射
    return np.exp(-hsp_distance / 10.0)


def predict_multisolvent_logS(peptide_hsp, peptide_charge, peptide_mw, n_residues, gravy):
    """
    为 8 种溶剂预测 logS。

    参数:
        peptide_hsp: dict {'d_d', 'd_p', 'd_h'}
        peptide_charge: 净电荷 pH7
        peptide_mw: 分子量
        n_residues: 残基数
        gravy: GRAVY 指数

    返回:
        list of dict: [{solvent_key, logS, category, ra_distance, score}, ...]
    """
    results = []

    for solvent_key, solvent_info in SOLVENTS.items():
        diel = solvent_info['dielectric']
        solv_hsp = {'d_d': solvent_info['d_d'],
                     'd_p': solvent_info['d_p'],
                     'd_h': solvent_info['d_h']}
        solv_class = solvent_info['class']

        # 1. HSP 距离
        ra = hsp_distance(peptide_hsp, solv_hsp)
        hsp_score = relative_solubility_score(ra)

        # 2. 经验修正

        # 电荷修正: 带电肽在极性/质子性溶剂中更易溶
        abs_charge = abs(peptide_charge)
        if solv_class == 'protic' and abs_charge > 0.5:
            charge_bonus = 0.5 * abs_charge  # 质子溶剂溶解带电肽
        elif solv_class == 'aprotic' and abs_charge > 0.5:
            charge_bonus = -0.3 * abs_charge  # 非质子溶剂不欢迎电荷
        else:
            charge_bonus = 0.0

        # 分子量修正: 大肽在 DMSO/NMP 中相对更好 (常用于合成)
        mw_effect = 0.0
        if solvent_key in ('dmso', 'nmp', 'dmf') and n_residues > 3:
            mw_effect = 0.3  # 多肽合成常用这些溶剂

        # GRAVY 修正: 疏水肽在 DCM 中偏好
        hydrophobicity_effect = 0.0
        if gravy > 0.5 and solvent_key == 'dcm':
            hydrophobicity_effect = 0.5  # DCM 溶解疏水肽好
        elif gravy < -1.0 and solvent_key == 'water':
            hydrophobicity_effect = 0.5  # 亲水肽水溶好

        # 3. 综合估算 logS (以水为基线, 调整到其他溶剂)
        # 水 logS ≈ -0.5*GRAVY - log10(N) - 0.3*|charge| - 2.0
        # 其他溶剂: 基于 HSP score 缩放
        base_logS_water = -0.5 * gravy - np.log10(max(n_residues, 1)) - \
                          0.3 * min(abs_charge, 2) - 2.0

        # HSP 距离 → logS 调整
        # Ra < 5: 溶解好于水; Ra 5-10: 相当; Ra > 10: 差于水
        if ra < 5:
            hsp_adjustment = 2.0 * (1 - ra / 5)  # up to +2 logS
        elif ra < 15:
            hsp_adjustment = -0.5 * (ra - 5) / 10  # up to -0.5
        else:
            hsp_adjustment = -1.0 - 0.5 * (ra - 15) / 15  # -1.0 to -2.0

        # 最终 logS
        logS = base_logS_water + hsp_adjustment + charge_bonus + mw_effect + \
               hydrophobicity_effect

        # 限制范围
        logS = max(-8.0, min(1.0, logS))

        # 分类
        if logS > -2.0:
            category = 'high'
        elif logS > -4.0:
            category = 'medium'
        else:
            category = 'low'

        results.append({
            'solvent': solvent_key,
            'solvent_name': solvent_info['name'],
            'logS': round(logS, 2),
            'category': category,
            'ra_distance': round(ra, 1),
            'hsp_score': round(hsp_score, 3),
            'hsp_distance_str': f"Ra={ra:.1f}",
        })

    # 按 HSP 得分降序排列 (得分越高越兼容)
    results.sort(key=lambda x: x['hsp_score'], reverse=True)
    return results
