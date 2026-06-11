"""
Pandas 模块2：文件读写
===========================
CSV, Excel, JSON 读写, 编码处理, 大文件分块读取。
"""

import pandas as pd
import os

SAVE_DIR = os.path.dirname(__file__)

# ============================================================
# 1. CSV 读写 (最常用)
# ============================================================
print("1. CSV 读写")
print("=" * 60)

df = pd.DataFrame({
    '序列':   ['AGK', 'ILVFW', 'KRDE', 'AKEFTL'],
    '分子量': [274.3, 676.9, 546.6, 707.8],
    'logS':  [-2.36, -4.14, -0.68, -2.80],
})

# 写入
csv_path = os.path.join(SAVE_DIR, 'demo.csv')
df.to_csv(csv_path, index=False)   # index=False 不写行号
print(f"   已保存: {csv_path}")

# 读入
df_read = pd.read_csv(csv_path)
print(f"   已读取:\n{df_read}\n")

# ---- 常用参数 ----
# 只读指定列
df_partial = pd.read_csv(csv_path, usecols=['序列', 'logS'])
print(f"   只读两列:\n{df_partial}\n")

# ============================================================
# 2. CSV 编码与分隔符
# ============================================================
print("2. 中文 CSV 编码处理")
print("=" * 60)

cn_df = pd.DataFrame({
    '药品': ['阿司匹林', '布洛芬', '咖啡因'],
    'MW':  [180.2, 206.3, 194.2],
})
cn_path = os.path.join(SAVE_DIR, 'cn_demo.csv')
# utf-8-sig 防止 Excel 打开乱码
cn_df.to_csv(cn_path, index=False, encoding='utf-8-sig')
print(f"   已保存中文 CSV: {cn_path}")

cn_read = pd.read_csv(cn_path)
print(f"   读取中文:\n{cn_read}\n")

# TSV (tab分隔) 读写
tsv_path = os.path.join(SAVE_DIR, 'demo.tsv')
df.to_csv(tsv_path, sep='\t', index=False)
tsv_df = pd.read_csv(tsv_path, sep='\t')
print(f"   TSV 读回: {tsv_df.shape[0]} 行")
print()

# ============================================================
# 3. Excel 读写
# ============================================================
print("3. Excel 读写")
print("=" * 60)

xlsx_path = os.path.join(SAVE_DIR, 'demo.xlsx')

try:
    # 写入 (可写多个 sheet)
    with pd.ExcelWriter(xlsx_path) as writer:
        df.to_excel(writer, sheet_name='多肽', index=False)
        cn_df.to_excel(writer, sheet_name='药品', index=False)
    print(f"   已保存: {xlsx_path} (2个sheet)")

    # 读取
    df_xl = pd.read_excel(xlsx_path, sheet_name='多肽')
    print(f"   读取 sheet '多肽':\n{df_xl}\n")

    # 查看所有 sheet 名
    xl = pd.ExcelFile(xlsx_path)
    print(f"   所有 sheet: {xl.sheet_names}")
except ImportError:
    print("   (openpyxl 未安装, 跳过 Excel)")
except Exception as e:
    print(f"   Excel 错误: {e}")
print()

# ============================================================
# 4. 大文件分块读取
# ============================================================
print("4. 大文件分块读取 (chunksize)")
print("=" * 60)

# chunksize 返回迭代器, 每次读 N 行
print("   分块读取大文件 (演示):")
chunk_count = 0
for chunk in pd.read_csv(csv_path, chunksize=2):
    chunk_count += 1
    print(f"   块 {chunk_count}: {chunk.shape[0]} 行")
print()

# ============================================================
# 5. 常用导出格式速查
# ============================================================
print("5. 格式速查")
print("=" * 60)

print("""
   df.to_csv('f.csv', index=False)         写CSV
   df.to_excel('f.xlsx', sheet_name='S1')  写Excel
   df.to_json('f.json')                    写JSON
   df.to_dict()                            转Python字典
   df.to_numpy()                           转NumPy数组
   pd.read_csv('f.csv')                    读CSV
   pd.read_excel('f.xlsx', sheet_name='S1')读Excel
   pd.read_json('f.json')                  读JSON
""")

# 清理 (Windows 下 Excel 文件可能被占用)
for f in ['demo.csv', 'demo.tsv', 'demo.xlsx', 'cn_demo.csv']:
    fp = os.path.join(SAVE_DIR, f)
    try:
        if os.path.exists(fp):
            os.remove(fp)
    except PermissionError:
        pass

print("[OK] Pandas 模块2 完成!")
