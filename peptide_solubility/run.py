"""
run.py — 多肽多溶剂溶解度预测工具
====================================
用法:
    python run.py                                    # 交互式
    python run.py AGK                                # 纯序列
    python run.py 'AGK[SMILES]LFP'                   # []插入片段
    python run.py 'OC(=O)CC' --seq AGK               # 片段+序列
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peptide_solubility.predict import predict_solubility
from peptide_solubility.utils import (
    parse_input, parse_bracketed_input, assemble_molecule,
    combine_fragment_peptide
)


def smart_predict(text):
    """
    智能预测: 自动检测输入格式。
    1. [] 格式 → 多段拼接
    2. 纯序列 → 直接用
    3. 纯 SMILES → 直接用
    """
    # 1. 检测 {} 格式 (非天然AA用花括号包裹)
    if '{' in text and '}' in text:
        segs = parse_bracketed_input(text)
        if len(segs) > 1:
            full_smi = assemble_molecule(segs)
            if full_smi:
                print(f"\n  解析为 {len(segs)} 段: "
                      f"{' + '.join(s['type']+':'+s['val'][:15] for s in segs)}")
                predict_solubility(full_smi, solvent='all', verbose=True)
                return
            else:
                print("  多段拼接失败, 回退...")

    # 过滤误输入的命令行
    if any(kw in text for kw in (':\\', '.exe', '.py', 'python')):
        print(f"  看起来是命令行不是多肽序列, 跳过")
        return

    # 2. 普通解析
    t, c = parse_input(text)
    if t == 'sequence':
        predict_solubility(c, solvent='all', verbose=True)
    elif t == 'smiles':
        predict_solubility(c, solvent='all', verbose=True)
    else:
        print(f"  无法解析: {text}")


def main():
    args = sys.argv[1:]

    # 命令行模式
    if args:
        # --frag + --seq 模式
        fragment = None
        sequence = None
        others = []

        i = 0
        while i < len(args):
            if args[i] in ('--frag', '-f') and i + 1 < len(args):
                fragment = args[i + 1]
                i += 2
            elif args[i] in ('--seq', '-s') and i + 1 < len(args):
                sequence = args[i + 1]
                i += 2
            else:
                others.append(args[i])
                i += 1

        if fragment and sequence:
            full_smi = combine_fragment_peptide(fragment, sequence)
            if full_smi:
                print(f"  片段: {fragment}")
                print(f"  序列: {sequence}")
                predict_solubility(full_smi, solvent='all', verbose=True)
            elif sequence:
                predict_solubility(sequence, solvent='all', verbose=True)
            return

        # 直接输入 (支持 [] 格式)
        for text in others:
            smart_predict(text)
        return

    # 交互式模式
    print("=" * 60)
    print("  多肽多溶剂溶解度预测工具")
    print("=" * 60)
    print()
    print("  输入格式:")
    print("    纯序列:     AGK 或 ALA-GLY-LYS")
    print("    纯SMILES:   C[C@H](N)C(=O)...")
    print("    {}插入片段: AGK{SMILES}LFP    ({}中放非天然AA的SMILES)")
    print()
    print("  输入 'q' 退出, 'all' 看示例")
    print()

    examples = [
        "AGK",
        "ILVFW",
        "KRDE",
        "AGK{N[C@@H](C(=O)O)c1ccccc1}LFP",
        "{OC(=O)CCCCCN1C(=O)C2C3C=CC3C2C1=O}SLAFQLLEQLIQQR",
    ]

    try:
        text = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if text.lower() == 'all':
        print("\n内置示例:\n")
        for ex in examples:
            print(f"  {ex}")
        return

    if text:
        smart_predict(text)


if __name__ == '__main__':
    main()
