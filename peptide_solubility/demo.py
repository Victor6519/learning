"""
demo.py — 多肽多溶剂溶解度预测演示
=======================================
测试多种多肽在 8 种溶剂中的溶解度对比。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peptide_solubility.predict import predict_solubility

# ================================================================
# 测试用例
# ================================================================
test_cases = [
    ("三肽 AGK (亲水+带电)",     "ALA-GLY-LYS"),
    ("五肽 疏水",                "ILE-LEU-VAL-PHE-TRP"),
    ("四肽 带电",                "LYS-ARG-ASP-GLU"),
    ("六肽 混合",                "ALA-LYS-GLU-PHE-THR-LEU"),
    ("十肽 重复序列",            "ALA-GLY-ALA-GLY-ALA-GLY-ALA-GLY-ALA-GLY"),
]

for desc, peptide in test_cases:
    print(f"\n{'='*75}")
    print(f"  {desc}")
    predict_solubility(peptide, solvent='all', verbose=True)

# ================================================================
# 汇总对比表
# ================================================================
print(f"\n\n{'='*75}")
print(f"  汇总: 各肽在各溶剂中的 logS")
print(f"{'='*75}")

# 收集所有结果
all_summaries = []
for desc, peptide in test_cases:
    results = predict_solubility(peptide, solvent='all', verbose=False)
    all_summaries.append((desc[:12], results))

# 表头
solvents_short = ['H2O', 'DMF', 'DCM', 'NMP', 'DMSO', 'MeOH', 'EtOH', 'MeCN']
print(f"  {'肽':12s} " + " ".join(f"{s:>8s}" for s in solvents_short))
print(f"  {'-'*80}")
for desc, results in all_summaries:
    logs = {r['solvent']: r['logS'] for r in results}
    row = f"  {desc:12s} " + " ".join(
        f"{logs.get(s, 0):8.2f}" for s in
        ['water', 'dmf', 'dcm', 'nmp', 'dmso', 'methanol', 'ethanol', 'acetonitrile']
    )
    print(row)

print(f"\n{'='*75}")
print(f"  图例: logS > -2 = high | -2~-4 = medium | < -4 = low")
print(f"  HSP 理论: Ra (溶剂-溶质距离) 越小 → 溶解越好")
print(f"{'='*75}")

print("\n[OK] demo 完成!")
