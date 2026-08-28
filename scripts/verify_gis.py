# -*- coding: utf-8 -*-
"""GIS 彩蛋数据集验证：回归任务 + 12 全模型（含 XGBoost/LightGBM/CatBoost）。"""
import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path
from app.ml.train import run_pipeline, LogCapture

df = pd.read_csv(r"D:\ML_help\demo-data\soil_gis.csv")
print("数据:", df.shape, "列:", list(df.columns))

log = LogCapture()
out_dir = Path(r"D:\ML_help\outputs\verify_gis")
t0 = time.time()
result = run_pipeline(
    df, target_col="yield", task_type="regression",
    id_cols=["site_id"], out_dir=out_dir, source_file=r"D:\ML_help\demo-data\soil_gis.csv",
    log=log,
)
elapsed = time.time() - t0
print(f"耗时 {elapsed:.1f}s")
print("最优模型:", result["best_model"]["name"], "R2=", result["best_model"]["metrics"]["r2"])
print("训练模型数:", len(result["models"]))
print("模型列表:", [m["name"] for m in result["models"]])
print("产物:", sorted(p.name for p in out_dir.iterdir()))
print("--- 日志尾 6 行 ---")
for line in log.lines[-6:]:
    print("  ", line)
