# -*- coding: utf-8 -*-
"""验证自动生成的复现脚本可独立运行（模拟学生下载后本地复现）。"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TASK = "631ab090ec08"
OUT = ROOT / "outputs" / TASK

# 1. 把示例数据复制到任务目录（模拟学生下载了 script.py + 数据）
data_src = ROOT / "sample_data" / "学生成绩示例.csv"
data_dst = OUT / "复现数据.csv"
data_dst.write_bytes(data_src.read_bytes())

# 2. 把 script.py 中的 DATA_PATH 改为任务目录下的数据
script_path = OUT / "script.py"
s = script_path.read_text(encoding="utf-8")
s = re.sub(r'DATA_PATH = r"[^"]*"', lambda m: 'DATA_PATH = r"' + str(data_dst) + '"', s)
script_path.write_text(s, encoding="utf-8")

# 3. 在任务目录运行复现脚本
env = dict(__import__("os").environ, PYTHONIOENCODING="utf-8")
p = subprocess.run(
    [sys.executable, str(script_path)],
    cwd=str(OUT), capture_output=True, text=True, encoding="gbk", errors="replace",
    timeout=300,
)
print(p.stdout)
if p.returncode != 0:
    print("STDERR:", p.stderr)
    sys.exit(1)

# 4. 检查产物
for f in ("confusion_matrix.png", "feature_importance.png", "report.md"):
    fp = OUT / f
    print(f"{f}: {'OK' if fp.is_file() and fp.stat().st_size > 0 else 'MISSING'}")
print("复现脚本独立运行验证通过")
