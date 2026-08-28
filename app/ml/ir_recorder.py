# -*- coding: utf-8 -*-
"""
Pipeline 中间表示记账（规格书 §5.4 核心模块 2）。

在训练过程中记录完整的 pipeline 信息，为源码导出提供数据基础。
原则：源码导出不是"训练完再反推"，而是边训练边记。
输出 storage/jobs/{task_id}/pipeline_ir.json
"""
import datetime
import json
import platform
import sys
from pathlib import Path


def _version(mod_name):
    """安全获取模块版本号。"""
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unknown"


def _environment():
    """记录运行环境版本号。"""
    return {
        "python_version": platform.python_version(),
        "sklearn_version": _version("sklearn"),
        "pandas_version": _version("pandas"),
        "numpy_version": _version("numpy"),
        "matplotlib_version": _version("matplotlib"),
        "platform": sys.platform,
    }


def new_ir(task_id: str, source_file: str, target_column: str,
           task_type: str, model_set: list, sort_metric: str,
           fold: int, session_id: int) -> dict:
    """创建初始 pipeline_ir 结构（训练开始前调用）。"""
    return {
        "task_id": task_id,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "data_info": {
            "source_file": source_file or "",
            "row_count": None,
            "col_count": None,
            "target_column": target_column,
            "feature_columns": [],
            "dropped_columns": [],
            "imputed_columns": {},
            "encoded_columns": {},
        },
        "setup_config": {
            "target": target_column,
            "train_size": 0.7,
            "fold_strategy": "stratifiedkfold" if task_type == "classification" else "kfold",
            "fold": fold,
            "session_id": session_id,
            "normalize": True,
            "transformation": False,
            "remove_multicollinearity": False,
            "multicollinearity_threshold": 0.9,
            "sort_metric": sort_metric,
            "task_type": task_type,
            "model_set": model_set,
        },
        "model_results": {
            "best_model_name": None,
            "best_model_params": {},
            "all_models_trained": [],
            "sort_metric": sort_metric,
            "cv_fold": fold,
        },
        "environment": _environment(),
    }


def update_data_info(ir: dict, df, dropped_columns: list,
                     imputed_columns: dict = None,
                     encoded_columns: dict = None) -> dict:
    """训练读取数据后记录数据规模与列信息。"""
    ir["data_info"]["row_count"] = int(len(df))
    ir["data_info"]["col_count"] = int(df.shape[1])
    ir["data_info"]["feature_columns"] = [str(c) for c in df.columns]
    ir["data_info"]["dropped_columns"] = [str(c) for c in (dropped_columns or [])]
    if imputed_columns:
        ir["data_info"]["imputed_columns"] = imputed_columns
    if encoded_columns:
        ir["data_info"]["encoded_columns"] = encoded_columns
    return ir


def update_model_results(ir: dict, results: list, best_name: str,
                         best_params: dict, sort_metric: str) -> dict:
    """训练完成后记录模型对比结果。"""
    ir["model_results"]["best_model_name"] = best_name
    ir["model_results"]["best_model_params"] = {str(k): str(v) for k, v in (best_params or {}).items()}
    ir["model_results"]["all_models_trained"] = [str(r["name"]) for r in results]
    ir["model_results"]["sort_metric"] = sort_metric
    return ir


def save_ir(ir: dict, out_dir) -> Path:
    """将 pipeline_ir 落盘到任务目录。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pipeline_ir.json"
    path.write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
