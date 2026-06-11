"""
predict.py — 统一预测入口
====================

用法:
    from peptide_solubility import predict_solubility

    # 单溶剂
    result = predict_solubility("ALA-GLY-LYS", solvent="water")
    result = predict_solubility("AGK", solvent="dmf")

    # 所有溶剂
    results = predict_solubility("ILE-LEU-VAL-PHE-TRP", solvent="all")
"""

from .utils import parse_input
from .features import extract_all_features
from .solvents import estimate_peptide_hsp, predict_multisolvent_logS


def predict_solubility(input_text, solvent='all', method='auto', verbose=False):
    """
    预测多肽溶解度。

    参数:
        input_text: SMILES 或氨基酸序列
        solvent:   'all' | 'water' | 'dmf' | 'dcm' | 'nmp' | 'dmso' |
                   'methanol' | 'ethanol' | 'acetonitrile'
        method:    'auto' | 'rule' | 'ml'
        verbose:   是否打印详细信息

    返回:
        单溶剂时: dict  {logS, category, solvent_name, ...}
        solvent='all'时: list of dict (8 种溶剂, 按溶解度排序)
    """
    # 1. 解析输入
    input_type, canonical = parse_input(input_text)
    if canonical is None:
        err = {'error': f'无法解析输入: {input_text}'}
        return [err] if solvent == 'all' else err

    # 2. 提取特征
    feats, names = extract_all_features(canonical, input_type)
    feat_dict = dict(zip(names, feats.tolist()))

    # 3. 获取关键参数
    n_residues = feat_dict.get('N_Residues', 1)
    peptide_mw = feat_dict.get('Seq_MW', feat_dict.get('MolWt', 200))
    charge = feat_dict.get('Net_Charge_pH7', 0)
    gravy = feat_dict.get('GRAVY', 0)

    # 4. 估计肽 HSP
    if input_type == 'sequence' or input_type is None:
        # 确保 canonical 是序列 (用于 HSP 估算)
        if input_type == 'smiles':
            from .utils import smiles_to_sequence
            seq = smiles_to_sequence(canonical)
            p_hsp = estimate_peptide_hsp(seq) if seq else estimate_peptide_hsp('A' * int(n_residues))
        else:
            p_hsp = estimate_peptide_hsp(canonical)
    else:
        from .utils import smiles_to_sequence
        seq = smiles_to_sequence(canonical)
        p_hsp = estimate_peptide_hsp(seq) if seq else estimate_peptide_hsp('A' * max(int(n_residues), 1))

    # 5. 多溶剂预测
    all_results = predict_multisolvent_logS(
        p_hsp, charge, peptide_mw, int(n_residues), gravy
    )

    # 6. 附加额外信息到每个结果
    for r in all_results:
        r['input_type'] = input_type
        r['canonical'] = canonical
        r['method'] = 'rule' if method != 'ml' else 'ml'
        r['peptide_hsp'] = p_hsp

    # 7. 返回
    if solvent == 'all':
        if verbose:
            _print_multisolvent(input_text, input_type, canonical, all_results)
        return all_results

    # 单溶剂查找
    from .solvents import get_solvent
    solv_key, solv_info = get_solvent(solvent)
    if solv_key is None:
        return {'error': f'未知溶剂: {solvent}'}

    for r in all_results:
        if r['solvent'] == solv_key:
            result = r
            break
    else:
        result = all_results[0]  # fallback

    if verbose:
        print(f"输入: {input_text}")
        print(f"类型: {input_type} -> {canonical}")
        print(f"溶剂: {solv_info['name']}")
        print(f"预测 logS: {result['logS']:.2f} log(mol/L)")
        print(f"分类: {result['category']}")
        print(f"HSP 距离: Ra={result['ra_distance']:.1f}")

    return result


def _print_multisolvent(input_text, input_type, canonical, results):
    """打印多溶剂对比表"""
    print(f"\n{'='*75}")
    print(f"  多肽: {input_text}")
    print(f"  类型: {input_type} -> {canonical}")
    print(f"{'='*75}")
    print(f"  {'溶剂':15s} {'logS':>8s} {'溶解度':>6s} {'Ra':>8s} {'HSP得分':>8s}")
    print(f"  {'-'*50}")
    for r in results:
        bar = _solubility_bar(r['logS'])
        print(f"  {r['solvent_name']:15s} {r['logS']:8.2f} {r['category']:>6s} "
              f"{r['ra_distance']:8.1f} {r['hsp_score']:8.3f} {bar}")

    best = results[0]
    worst = results[-1]
    print(f"  {'-'*50}")
    print(f"  >>> 最佳溶剂: {best['solvent_name']} (HSP={best['hsp_score']:.3f}, logS={best['logS']:.2f})")
    print(f"  >>> 最差溶剂: {worst['solvent_name']} (HSP={worst['hsp_score']:.3f}, logS={worst['logS']:.2f})")


def _solubility_bar(logS):
    """简易可视化条"""
    width = max(0, min(20, int((logS + 8) * 2.5)))
    return '|' + '#' * width + ' ' * (20 - width) + '|'
