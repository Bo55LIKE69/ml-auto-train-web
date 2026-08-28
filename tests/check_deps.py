# -*- coding: utf-8 -*-
"""检查规格书所需依赖是否可用。"""
import importlib.util

mods = ["pycaret", "docx", "flaml", "openpyxl", "jinja2", "fastapi", "pandas", "sklearn", "matplotlib", "uvicorn"]
for m in mods:
    print(f"{m:12s} {'OK' if importlib.util.find_spec(m) else 'MISSING'}")
