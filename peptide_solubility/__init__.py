"""
peptide_solubility — 多肽溶解度预测工具库
=============================================
支持 SMILES 和氨基酸序列两种输入，
结合 RDKit 描述符和肽专属特征做溶解度预测。
"""

from .predict import predict_solubility
from .utils import parse_input, sequence_to_smiles, smiles_to_sequence

__version__ = "0.1.0"
