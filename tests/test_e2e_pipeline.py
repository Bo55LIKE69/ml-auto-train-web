# -*- coding: utf-8 -*-
"""端到端流水线测试：用 sample_data 学生成绩示例数据走完整训练。
运行：cd D:/ML_help && .venv/Scripts/python tests/test_e2e_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.explore import read_table
from app.ml.train import run_pipeline

DATA = Path(__file__).resolve().parent.parent / "sample_data" / "学生成绩示例.csv"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "e2e_demo"

if __name__ == "__main__":
    df = read_table(DATA)
    print(f"数据读取：{df.shape[0]} 行 x {df.shape[1]} 列")

    result = run_pipeline(
        df=df,
        target_col="是否通过",
        task_type="auto",
        id_cols=None,
        out_dir=OUT,
        source_file=str(DATA),
    )

    print(f"\n任务类型: {result['task_type']}")
    print(f"样本数: {result['n_samples']}, 剔除列: {result['drop_cols']}")
    print(f"警告: {result['warnings']}")
    print("\n模型对比:")
    for m in result["models"]:
        print(f"  {m['name']:8s} {m['metrics']}")
    print(f"\n最优模型: {result['best_model']}")
    print(f"特征重要性 Top5: {result['feature_importance'][:5]}")
    print(f"图表: {list(result['plots'].keys())}")
    print(f"\n产物目录: {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  - {f.name} ({f.stat().st_size} bytes)")
    print("\n✅ 端到端流水线测试通过")
