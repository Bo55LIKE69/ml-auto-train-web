# -*- coding: utf-8 -*-
"""端到端验证：新引擎（12模型+Word报告+IR记账）跑通学生成绩数据。"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path
from app.ml.train import run_pipeline, LogCapture

df = pd.read_csv(r"D:\ML_help\sample_data\学生成绩示例.csv", encoding="utf-8")
print("数据:", df.shape, "列:", list(df.columns))

log = LogCapture()
out_dir = Path(r"D:\ML_help\outputs\verify_v2")
result = run_pipeline(
    df, target_col="是否通过", task_type="classification",
    id_cols=["学号"], out_dir=out_dir, source_file=r"D:\ML_help\sample_data\学生成绩示例.csv",
    model_set=["lr", "knn", "nb", "dt", "rf", "et", "ada", "gbc", "svm"],
    log=log,
)
print("最优模型:", result["best_model"]["name"], result["best_model"]["metrics"]["f1"])
print("训练模型数:", len(result["models"]))
print("产物:", sorted(p.name for p in out_dir.iterdir()))
print("--- 日志前 8 行 ---")
for line in log.lines[:8]:
    print(line)
print("--- 日志后 5 行 ---")
for line in log.lines[-5:]:
    print(line)
