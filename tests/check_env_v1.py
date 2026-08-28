# -*- coding: utf-8 -*-
"""检查 v1.0.0 升级所需依赖与环境"""
import importlib.util
import os
import shutil
import subprocess
import sys

print("=== Python ===", sys.version.split()[0])

mods = ["shap", "docx2pdf", "reportlab", "pypdf", "jinja2", "docx", "sklearn",
        "xgboost", "lightgbm", "catboost", "fastapi", "pandas", "matplotlib"]
for m in mods:
    spec = importlib.util.find_spec(m)
    print(f"  {m}: {'OK' if spec else 'MISSING'}")

# LibreOffice
soffice = shutil.which("soffice") or shutil.which("soffice.exe")
lo_paths = [r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"]
found = soffice or next((p for p in lo_paths if os.path.exists(p)), None)
print("  LibreOffice:", found or "NOT FOUND")

# git
print("=== git ===")
print(subprocess.run(["git", "-C", "D:/ML_help", "status", "--short"],
                     capture_output=True, text=True, encoding="utf-8", errors="replace").stdout or "(clean)")
print("=== git remote ===")
print(subprocess.run(["git", "-C", "D:/ML_help", "remote", "-v"],
                     capture_output=True, text=True, encoding="utf-8", errors="replace").stdout)

# 服务状态
import urllib.request
try:
    r = urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=3)
    print("=== server ===", r.status)
except Exception as e:
    print("=== server === DOWN:", e)
