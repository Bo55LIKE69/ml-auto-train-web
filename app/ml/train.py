# -*- coding: utf-8 -*-
"""
多模型训练业务模块（流水线核心）：
    预处理 → 7:3 划分 → 批量训练基础模型 → 指标评估 → 选出最优模型
    → 绘图（混淆矩阵/散点图/特征重要性）→ 生成 Markdown 报告与可复现脚本

分类候选：逻辑回归 / 随机森林 / 决策树 / SVM
回归候选：线性回归 / 随机森林回归 / 决策树回归
（遵循"先跑基础模型"原则，不直接上复杂大模型）
"""
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.config import RANDOM_STATE, TEST_SIZE
from app.ml.evaluate import evaluate_classification, evaluate_regression
from app.ml.plots import (plot_confusion_matrix, plot_feature_importance,
                          plot_scatter)
from app.ml.preprocess import prepare_dataset
from app.ml.report import generate_report, generate_script

# ---- 基础模型候选集（全部默认参数起步，保证可复现）----
CLASSIFIERS = {
    "逻辑回归": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "随机森林": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "决策树": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
}

REGRESSORS = {
    "线性回归": LinearRegression(),
    "随机森林回归": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "决策树回归": DecisionTreeRegressor(random_state=RANDOM_STATE),
}


def _encode_y(task_type, y):
    """
    根据任务类型编码目标列：
    - 分类：LabelEncoder，返回 (编码数组, 编码器, 类别名列表)
    - 回归：强制转 float（无法转换的置 NaN，由调用方剔除）
    """
    if task_type == "classification":
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)
        return y_enc, encoder, [str(c) for c in encoder.classes_]
    y_enc = pd.to_numeric(y, errors="coerce").astype(float).values
    return y_enc, None, None


def run_pipeline(df, target_col, task_type="auto", id_cols=None,
                 test_size=TEST_SIZE, random_state=RANDOM_STATE,
                 out_dir=None, source_file=None):
    """
    完整训练流水线入口。

    参数：
        df          原始 DataFrame
        target_col  目标标签列
        task_type   auto / classification / regression
        id_cols     显式剔除的 ID 列（前端多选）
        out_dir     产物输出目录（图表/报告/脚本），None 则不落盘
        source_file 原始数据文件路径（写入复现脚本用）

    返回：结果 dict（可直接 JSON 序列化，含模型对比、最优模型、图 URL 等）
    """
    out_dir = Path(out_dir) if out_dir else None

    # ---- 1. 预处理 ----
    X, y, _, preprocessor, meta = prepare_dataset(df, target_col, id_cols)
    if task_type not in ("auto", "classification", "regression"):
        raise ValueError(f"非法的 task_type: {task_type}")
    if task_type != "auto":
        meta["task_type"] = task_type          # 用户强制指定任务类型
    task_type = meta["task_type"]

    # ---- 2. 目标列编码 ----
    y_enc, y_encoder, class_names = _encode_y(task_type, y)
    if task_type == "regression":
        mask = ~np.isnan(y_enc)
        if not mask.all():
            meta["warnings"].append(f"目标列有 {int((~mask).sum())} 行无法转为数值，已剔除")
            X, y_enc = X[mask], y_enc[mask]

    # ---- 3. 7:3 划分（分类按标签分层抽样，保证类别比例一致）----
    stratify = y_enc if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=stratify)

    # ---- 4. 预处理管道 fit（只 fit 训练集，防数据泄漏）----
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = [str(n) for n in preprocessor.get_feature_names_out()]

    # ---- 5. 批量训练与评估 ----
    models = CLASSIFIERS if task_type == "classification" else REGRESSORS
    results = []
    model_refs, preds = {}, {}
    best_name, best_score = None, -float("inf")
    rf_model = None  # 特征重要性图固定使用随机森林（树模型自带可解释性）

    for name, model in models.items():
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)
        model_refs[name], preds[name] = model, y_pred

        if task_type == "classification":
            y_proba = model.predict_proba(X_test_t) if hasattr(model, "predict_proba") else None
            metrics, cm = evaluate_classification(y_test, y_pred, y_proba)
            score = metrics["f1"]                     # 分类选优指标：F1(macro)
        else:
            metrics = evaluate_regression(y_test, y_pred)
            cm = None
            score = metrics["r2"]                     # 回归选优指标：R²
        results.append({"name": name, "metrics": metrics})

        if score > best_score:
            best_name, best_score = name, score
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            rf_model = model

    best_metrics = next(r["metrics"] for r in results if r["name"] == best_name)
    best_pred = preds[best_name]

    # ---- 6. 特征重要性（随机森林 top15）----
    importance = []
    if rf_model is not None:
        imp = np.asarray(rf_model.feature_importances_)
        order = np.argsort(imp)[::-1][:15]
        importance = [
            {"feature": feature_names[i], "importance": round(float(imp[i]), 4)}
            for i in order
        ]

    # ---- 7. 产物落盘：图表 / 报告 / 复现脚本 ----
    task_id = out_dir.name if out_dir else uuid.uuid4().hex[:12]
    plot_files = {}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if task_type == "classification":
            p = out_dir / "confusion_matrix.png"
            plot_confusion_matrix(y_test, best_pred, class_names, p)
            plot_files["confusion_matrix"] = p.name
        else:
            p = out_dir / "scatter.png"
            plot_scatter(y_test, best_pred, p)
            plot_files["scatter"] = p.name
        if rf_model is not None:
            p = out_dir / "feature_importance.png"
            plot_feature_importance(rf_model, feature_names, p)
            plot_files["feature_importance"] = p.name

    result = {
        "task_id": task_id,
        "target_col": target_col,
        "task_type": task_type,
        "n_samples": int(meta["n_samples"]),
        "n_features_raw": int(meta["n_features_raw"]),
        "n_features_after_prep": len(feature_names),
        "class_names": class_names,
        "warnings": meta["warnings"],
        "drop_cols": meta["drop_cols"],
        "models": results,
        "best_model": {
            "name": best_name,
            "metrics": best_metrics,
            "reason": "F1(macro) 最高" if task_type == "classification" else "R² 最高",
        },
        "feature_importance": importance,
        "plots": {k: f"/api/download/{task_id}/plot/{v}" for k, v in plot_files.items()},
        "report_url": f"/api/download/{task_id}/report.md",
        "script_url": f"/api/download/{task_id}/script.py",
    }

    if out_dir is not None:
        # Markdown 实验报告
        report_md = generate_report(result, df, y, meta)
        (out_dir / "report.md").write_text(report_md, encoding="utf-8")
        # 可独立运行的复现脚本
        script_py = generate_script(result, source_file, target_col, task_type)
        (out_dir / "script.py").write_text(script_py, encoding="utf-8")
        # 结果 JSON（供 /api/result/{task_id} 查询）
        (out_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result
