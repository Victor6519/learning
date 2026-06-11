"""
RDKit 模块1：分子读写与基本信息
===================================
从 SMILES 创建分子对象，遍历原子/键/环，SDF 文件读写。
"""

from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors

PYTHON = r"c:/Users/k/miniconda3/envs/quant/python"

# ============================================================
# 1. 从 SMILES 创建分子对象
# ============================================================
# SMILES 是最常见的分子线性表示法
# MolFromSmiles 返回 rdkit.Chem.rdchem.Mol 对象（失败返回 None）

caffeine_smi = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
caffeine = Chem.MolFromSmiles(caffeine_smi)

print("1. 从 SMILES 创建分子")
print(f"   SMILES: {caffeine_smi}")
print(f"   分子对象: {caffeine}")
print(f"   类型: {type(caffeine)}")
print()

# ---- 多个分子一起创建 ----
smiles_list = {
    "阿司匹林": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "布洛芬":   "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "对乙酰氨基酚": "CC(=O)NC1=CC=C(C=C1)O",
    "咖啡因":   "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}
mols = {name: Chem.MolFromSmiles(smi) for name, smi in smiles_list.items()}

# ============================================================
# 2. 分子转回 SMILES
# ============================================================
# MolToSmiles 将分子对象标准化为 SMILES
# canonical=True 确保输出是规范形式（同一分子的不同输入 SMILES 输出一致）

print("2. 分子 → SMILES（规范化）")
for name, mol in mols.items():
    smi = Chem.MolToSmiles(mol, canonical=True)
    print(f"   {name}: {smi}")
print()

# ---- 不规范的 SMILES 输入 → 规范化输出 ----
# 同一分子的不同 SMILES 写法，规范化后变成同一个
smi_variants = [
    "CCO",       # 乙醇
    "OCC",       # 乙醇（另一种写法）
]
print("   同一分子不同 SMILES → 规范化:")
for s in smi_variants:
    m = Chem.MolFromSmiles(s)
    print(f"   {s} → {Chem.MolToSmiles(m)}")
print()

# ============================================================
# 3. 分子对象的属性遍历
# ============================================================
# 用阿司匹林做详细讲解
aspirin = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")

# 3a. 原子遍历
print("3a. 原子遍历（阿司匹林）")
print(f"   总原子数: {aspirin.GetNumAtoms()}")
print(f"   原子   序号   符号   原子序数   度   是否在环   explicit_H   implicit_H")
for atom in aspirin.GetAtoms():
    print(f"   {atom.GetSymbol():>4}  {atom.GetIdx():>4}  "
          f"{atom.GetSymbol():>4}  {atom.GetAtomicNum():>6}  "
          f"{atom.GetDegree():>4}  "
          f"{atom.IsInRing():>6}    {atom.GetNumExplicitHs():>8}  {atom.GetNumImplicitHs():>8}")
print()

# 3b. 键遍历
print("3b. 键遍历")
print(f"   总键数: {aspirin.GetNumBonds()}")
for bond in aspirin.GetBonds():
    a1 = bond.GetBeginAtomIdx()
    a2 = bond.GetEndAtomIdx()
    btype = bond.GetBondType()  # SINGLE, DOUBLE, TRIPLE, AROMATIC
    is_in_ring = bond.IsInRing()
    print(f"   {a1} — {a2}   {str(btype):>10}   {'ring' if is_in_ring else 'chain'}")
print()

# 3c. 环信息
# GetRingInfo() 返回分子中所有环的信息
print("3c. 环信息")
ri = aspirin.GetRingInfo()
print(f"   环的数量: {ri.NumRings()}")
for i in range(ri.NumRings()):
    ring_atoms = ri.AtomRings()[i]
    print(f"   环 {i+1}: 原子 {tuple(ring_atoms)}")
print(f"   每个原子所属环数: {[ri.NumAtomRings(i) for i in range(aspirin.GetNumAtoms())]}")
print()

# 3d. 获取环的 SMILES（使用 GetSymmSSSR 的辅助方法）
# SSSR = Smallest Set of Smallest Rings
sssr = Chem.GetSymmSSSR(aspirin)
print(f"   SSSR 环数: {len(sssr)}")
for i, ring in enumerate(sssr):
    print(f"   环 {i}: 原子 {tuple(ring)}")
print()

# ============================================================
# 4. 分子基础属性速查
# ============================================================
print("4. 分子基础属性速查")
mol = aspirin
print(f"   GetNumAtoms()    原子数:     {mol.GetNumAtoms()}")
print(f"   GetNumBonds()    键数:       {mol.GetNumBonds()}")
print(f"   GetNumHeavyAtoms()  重原子数: {mol.GetNumHeavyAtoms()}")
print(f"   GetRingInfo().NumRings() 环数: {mol.GetRingInfo().NumRings()}")
print(f"   CalcNumRotatableBonds()  可旋转键: {rdMolDescriptors.CalcNumRotatableBonds(mol)}")
print(f"   CalcNumHeteroatoms() 杂原子数: {rdMolDescriptors.CalcNumHeteroatoms(mol)}")
print()

# ============================================================
# 5. SDF 文件读写
# ============================================================
import os

# -- 写入 SDF --
sdf_path = os.path.join(os.path.dirname(__file__), "demo.sdf")
writer = Chem.SDWriter(sdf_path)
for name, mol in mols.items():
    mol.SetProp("_Name", name)       # 设置分子名称
    mol.SetProp("SOURCE", "rdkit_demo")  # 自定义属性
    writer.write(mol)
writer.close()
print(f"5. 写了 {len(mols)} 个分子到 {sdf_path}")

# -- 从 SDF 读取 --
supplier = Chem.SDMolSupplier(sdf_path)
print("   从 SDF 读取:")
for i, mol in enumerate(supplier):
    if mol is None:
        print(f"   分子 {i}: 读取失败")
        continue
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown"
    source = mol.GetProp("SOURCE") if mol.HasProp("SOURCE") else ""
    smi = Chem.MolToSmiles(mol)
    print(f"   [{i}] {name}: {smi}  (source={source})")

supplier = None  # 释放文件句柄

print("\n[OK] 模块1 完成!")
