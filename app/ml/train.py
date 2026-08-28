# -*- coding: utf-8 -*-
"""
多模型训练业务模块（流水线核心，规格书 §5.3）：
    预处理 → 5 折交叉验证 → 批量训练（12 模型）→ 指标评估 → 选出最优模型
    → 绘图（混淆矩阵/散点图/特征重要性）→ 生成报告（Word+Markdown）与可复现脚本

分类候选（12）：逻辑回归 / K近邻 / 朴素贝叶斯 / 决策树 / 随机森林 / 极端随机树 /
               AdaBoost / 梯度提升 / XGBoost / LightGBM / CatBoost / SVM
回归候选：线性回归 / K近邻 / 决策树 / 随机森林 / 极端随机树 / AdaBoost /
         梯度提升 / XGBoost / LightGBM / CatBoost
（遵循规格书附录 A 模型缩写对照表）
"""
import contextlib
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (cross_val_predict, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.config import (DEFAULT_FOLD, DEFAULT_MODEL_SET, RANDOM_STATE,
                        TEST_SIZE, TRAIN_TIMEOUT_SECONDS)
from app.ml.evaluate import evaluate_classification, evaluate_regression
from app.ml.ir_recorder import (new_ir, save_ir, update_data_info,
                                update_model_results)
from app.ml.plots import (plot_confusion_matrix, plot_feature_importance,
                          plot_scatter)
from app.ml.preprocess import prepare_dataset
from app.ml.report import generate_report, generate_script
from app.ml.word_report import generate_word_report

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # 未安装时降级
    XGBClassifier = XGBRegressor = None
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = LGBMRegressor = None
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = CatBoostRegressor = None


# ========== 模型目录（前端手动选模型展示用） ==========
MODEL_CATALOG = {
    "classification": {
        "lr":       {"name": "逻辑回归", "desc": "线性分类，训练快，可解释性强"},
        "knn":      {"name": "K近邻", "desc": "基于距离的实例学习"},
        "nb":       {"name": "朴素贝叶斯", "desc": "基于贝叶斯定理，适合文本/高维"},
        "dt":       {"name": "决策树", "desc": "树形规则，可解释性最好"},
        "rf":       {"name": "随机森林", "desc": "Bagging 集成，抗过拟合"},
        "et":       {"name": "极端随机树", "desc": "随机森林变体，更快更随机"},
        "ada":      {"name": "AdaBoost", "desc": "Boosting 集成，对弱分类器加权"},
        "gbc":      {"name": "梯度提升", "desc": "GBDT，sklearn 原生实现"},
        "xgboost":  {"name": "XGBoost", "desc": "经典 GBDT 加速实现（可选依赖）"},
        "lightgbm": {"name": "LightGBM", "desc": "轻量 GBDT，训练快（可选依赖）"},
        "catboost": {"name": "CatBoost", "desc": "类别特征友好（可选依赖）"},
        "svm":      {"name": "SVM", "desc": "支持向量机（线性核，避免慢）"},
    },
    "regression": {
        "lr":       {"name": "线性回归", "desc": "线性模型，可解释性强"},
        "knn":      {"name": "K近邻回归", "desc": "基于距离的实例学习"},
        "dt":       {"name": "决策树回归", "desc": "树形规则回归"},
        "rf":       {"name": "随机森林回归", "desc": "Bagging 集成回归"},
        "et":       {"name": "极端随机树回归", "desc": "随机森林变体回归"},
        "ada":      {"name": "AdaBoost回归", "desc": "Boosting 集成回归"},
        "gbc":      {"name": "梯度提升回归", "desc": "GBDT 回归"},
        "xgboost":  {"name": "XGBoost回归", "desc": "GBDT 回归（可选依赖）"},
        "lightgbm": {"name": "LightGBM回归", "desc": "轻量 GBDT 回归（可选依赖）"},
        "catboost": {"name": "CatBoost回归", "desc": "类别特征友好回归（可选依赖）"},
    },
}


def _build_models(task_type: str, model_set: list = None) -> dict:
    """
    按规格书附录 A 构建模型字典。
    model_set 为缩写列表（如 ["lr","rf","xgboost"]），None 表示全部可用。
    """
    rs = RANDOM_STATE
    clf = {}
    if "lr" in model_set:
        clf["逻辑回归"] = LogisticRegression(max_iter=2000, random_state=rs)
    if "knn" in model_set:
        clf["K近邻"] = KNeighborsClassifier(n_neighbors=5)
    if "nb" in model_set:
        clf["朴素贝叶斯"] = GaussianNB()
    if "dt" in model_set:
        clf["决策树"] = DecisionTreeClassifier(random_state=rs)
    if "rf" in model_set:
        clf["随机森林"] = RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1)
    if "et" in model_set:
        clf["极端随机树"] = ExtraTreesClassifier(n_estimators=200, random_state=rs, n_jobs=-1)
    if "ada" in model_set:
        clf["AdaBoost"] = AdaBoostClassifier(random_state=rs)
    if "gbc" in model_set:
        clf["梯度提升"] = GradientBoostingClassifier(random_state=rs)
    if "xgboost" in model_set and XGBClassifier is not None:
        clf["XGBoost"] = XGBClassifier(n_estimators=200, random_state=rs,
                                       eval_metric="logloss", verbosity=0)
    if "lightgbm" in model_set and LGBMClassifier is not None:
        clf["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=rs,
                                         verbose=-1)
    if "catboost" in model_set and CatBoostClassifier is not None:
        clf["CatBoost"] = CatBoostClassifier(n_estimators=200, random_state=rs,
                                             verbose=False, allow_writing_files=False)
    if "svm" in model_set:
        clf["SVM"] = SVC(probability=True, random_state=rs)
    return clf


def _build_regressors(model_set: list = None) -> dict:
    """回归模型集（规格书附录 A 回归部分）。"""
    rs = RANDOM_STATE
    reg = {}
    if "lr" in model_set:
        reg["线性回归"] = LinearRegression()
    if "knn" in model_set:
        reg["K近邻回归"] = KNeighborsRegressor(n_neighbors=5)
    if "dt" in model_set:
        reg["决策树回归"] = DecisionTreeRegressor(random_state=rs)
    if "rf" in model_set:
        reg["随机森林回归"] = RandomForestRegressor(n_estimators=200, random_state=rs, n_jobs=-1)
    if "et" in model_set:
        reg["极端随机树回归"] = ExtraTreesRegressor(n_estimators=200, random_state=rs, n_jobs=-1)
    if "ada" in model_set:
        reg["AdaBoost回归"] = AdaBoostRegressor(random_state=rs)
    if "gbc" in model_set:
        reg["梯度提升回归"] = GradientBoostingRegressor(random_state=rs)
    if "xgboost" in model_set and XGBRegressor is not None:
        reg["XGBoost回归"] = XGBRegressor(n_estimators=200, random_state=rs, verbosity=0)
    if "lightgbm" in model_set and LGBMRegressor is not None:
        reg["LightGBM回归"] = LGBMRegressor(n_estimators=200, random_state=rs, verbose=-1)
    if "catboost" in model_set and CatBoostRegressor is not None:
        reg["CatBoost回归"] = CatBoostRegressor(n_estimators=200, random_state=rs,
                                                verbose=False, allow_writing_files=False)
    return reg


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


class LogCapture:
    """
    捕获 sklearn 训练输出，支持增量读取（规格书 §5.3）。
    训练在后台线程执行时，前端通过 log_cursor 轮询增量日志。
    """
    def __init__(self):
        self.lines = []
        self._buffer = io.StringIO()

    def __enter__(self):
        import sys
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self._buffer
        sys.stderr = self._buffer
        return self

    def __exit__(self, *args):
        import sys
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        self.lines = self._buffer.getvalue().splitlines()

    def get_lines_since(self, cursor: int = 0) -> list:
        """返回第 cursor 行之后的新增日志（用于轮询）。"""
        return self.lines[cursor:]

    def append(self, msg: str):
        """直接追加一条日志（不经过 stdout 重定向时用）。"""
        self._buffer.write(msg + "\n")
        self.lines = self._buffer.getvalue().splitlines()


def run_pipeline(df, target_col, task_type="auto", id_cols=None,
                 test_size=TEST_SIZE, random_state=RANDOM_STATE,
                 out_dir=None, source_file=None, model_set=None,
                 fold=DEFAULT_FOLD, timeout=TRAIN_TIMEOUT_SECONDS,
                 log=None, progress_cb=None):
    """
    完整训练流水线入口（规格书 §5.3）。

    参数：
        df           原始 DataFrame
        target_col   目标标签列
        task_type    auto / classification / regression
        id_cols      显式剔除的 ID 列（前端多选）
        out_dir      产物输出目录（图表/报告/脚本），None 则不落盘
        source_file  原始数据文件路径（写入复现脚本用）
        model_set    模型缩写列表，None 用 DEFAULT_MODEL_SET
        fold         CV 折数
        timeout      单任务超时秒数
        log          LogCapture 实例（收集训练日志）
        progress_cb  进度回调 (stage, percent, message)

    返回：结果 dict（可直接 JSON 序列化，含模型对比、最优模型、图 URL 等）
    """
    t_start = time.time()
    out_dir = Path(out_dir) if out_dir else None
    log = log or LogCapture()
    model_set = list(model_set) if model_set else list(DEFAULT_MODEL_SET)

    def _progress(stage, pct, msg):
        log.append(f"[{stage}] {msg}")
        if progress_cb:
            progress_cb(stage, pct, msg)

    # ---- 1. 预处理 ----
    X, y, _, preprocessor, meta = prepare_dataset(df, target_col, id_cols)
    if task_type not in ("auto", "classification", "regression"):
        raise ValueError(f"非法的 task_type: {task_type}")
    if task_type != "auto":
        meta["task_type"] = task_type          # 用户强制指定任务类型
    task_type = meta["task_type"]
    _progress("preprocess", 5, f"预处理完成：{meta['n_samples']} 样本 x {meta['n_features_raw']} 特征，任务类型={task_type}")

    # ---- 2. 目标列编码 ----
    y_enc, y_encoder, class_names = _encode_y(task_type, y)
    if task_type == "regression":
        mask = ~np.isnan(y_enc)
        if not mask.all():
            meta["warnings"].append(f"目标列有 {int((~mask).sum())} 行无法转为数值，已剔除")
            X, y_enc = X[mask], y_enc[mask]
    _progress("encode", 10, "目标列编码完成")

    # ---- 3. 数据划分（分类按标签分层抽样）----
    stratify = y_enc if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=stratify)

    # ---- 4. 预处理管道 fit（只 fit 训练集，防数据泄漏）----
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = [str(n) for n in preprocessor.get_feature_names_out()]
    _progress("split", 15, f"划分完成：训练集 {X_train_t.shape[0]}，测试集 {X_test_t.shape[0]}，预处理后 {len(feature_names)} 维")

    # ---- 5. 批量训练（规格书：超时降级）----
    models = (_build_models(task_type, model_set)
              if task_type == "classification" else _build_regressors(model_set))
    if not models:
        raise ValueError("没有可用的模型（所选模型集为空或依赖未安装）")

    results = []
    model_refs, preds = {}, {}
    best_name, best_score = None, -float("inf")
    rf_model = None
    total = len(models)

    for i, (name, model) in enumerate(models.items(), 1):
        if time.time() - t_start > timeout:
            log.append(f"! 训练超时（>{timeout}s），中止剩余模型")
            break
        _progress("train", 15 + int(60 * (i - 1) / max(total, 1)),
                  f"训练模型 {i}/{total}：{name}")
        t_m = time.time()
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)
        model_refs[name], preds[name] = model, y_pred

        if task_type == "classification":
            y_proba = model.predict_proba(X_test_t) if hasattr(model, "predict_proba") else None
            metrics, cm = evaluate_classification(y_test, y_pred, y_proba)
            score = metrics["f1"]
        else:
            metrics = evaluate_regression(y_test, y_pred)
            cm = None
            score = metrics["r2"]
        metrics["train_time_s"] = round(time.time() - t_m, 2)
        results.append({"name": name, "metrics": metrics})
        _score_key = "F1" if task_type == "classification" else "R2"
        log.append(f"    {name:14s} -> {_score_key}={metrics.get('f1', metrics.get('r2'))} 耗时 {metrics['train_time_s']}s")

        if score > best_score:
            best_name, best_score = name, score
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            rf_model = model

    if best_name is None:
        raise RuntimeError("所有模型均训练失败")

    best_metrics = next(r["metrics"] for r in results if r["name"] == best_name)
    best_pred = preds[best_name]
    _progress("train", 80, f"最优模型：{best_name}（{best_score}）")

    # ---- 6. 特征重要性（随机森林 top15）----
    importance = []
    if rf_model is not None:
        imp = np.asarray(rf_model.feature_importances_)
        order = np.argsort(imp)[::-1][:15]
        importance = [
            {"feature": feature_names[i], "importance": round(float(imp[i]), 4)}
            for i in order
        ]

    # ---- 7. 产物落盘：图表 / 报告 / 复现脚本 / pipeline_ir ----
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

        # 指标对比图（规格书第四章要求）
        from app.ml.plots import plot_metrics_comparison
        p = out_dir / "metrics_comparison.png"
        plot_metrics_comparison(results, task_type, p)
        plot_files["metrics_comparison"] = p.name

        # 相关性热力图（数值特征两两相关性）
        from app.ml.plots import plot_correlation_heatmap
        try:
            if plot_correlation_heatmap(df, p := out_dir / "correlation.png",
                                         max_features=30):
                plot_files["correlation"] = p.name
        except Exception as e:
            log.append(f"! 相关性热力图生成失败：{e}")

        # SHAP 可解释性图（最优模型，采样控制耗时）
        from app.ml.plots import plot_shap_summary
        try:
            from app.config import SHAP_MAX_SAMPLES
            n_shap = min(len(X_test_t), SHAP_MAX_SAMPLES)
            import numpy as _np
            X_shap = X_test_t[:n_shap] if hasattr(X_test_t, "__getitem__") else _np.asarray(X_test_t)[:n_shap]
            if plot_shap_summary(model_refs[best_name], X_shap, feature_names,
                                 out_dir / "shap_summary.png"):
                plot_files["shap_summary"] = "shap_summary.png"
                log.append("[shap] SHAP 特征重要性图已生成 shap_summary.png")
        except Exception as e:
            log.append(f"! SHAP 图生成失败（已跳过）：{e}")

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
        "plots": {k: f"/api/download/{task_id}/{v}" for k, v in plot_files.items()},
        "report_url": f"/api/download/{task_id}/report.md",
        "docx_url": f"/api/download/{task_id}/report.docx",
        "script_url": f"/api/download/{task_id}/script.py",
        "training_time_total_seconds": round(time.time() - t_start, 2),
        "fold_count": fold,
        "session_id": random_state,
    }

    if out_dir is not None:
        # pipeline_ir 中间表示记账（规格书核心模块2）
        sort_metric = "F1" if task_type == "classification" else "R2"
        ir = new_ir(task_id, source_file, target_col, task_type, model_set,
                    sort_metric, fold, random_state)
        ir = update_data_info(ir, df, meta["drop_cols"])
        best_params = {}
        if best_name in model_refs and hasattr(model_refs[best_name], "get_params"):
            best_params = model_refs[best_name].get_params()
        ir = update_model_results(ir, results, best_name, best_params, sort_metric)
        save_ir(ir, out_dir)

        # Markdown 实验报告
        report_md = generate_report(result, df, y, meta)
        (out_dir / "report.md").write_text(report_md, encoding="utf-8")
        # Word 实验报告（规格书价值锚点，七章 docx）
        try:
            generate_word_report(result, df, y, meta, out_dir / "report.docx")
            log.append("[report] Word 七章报告已生成 report.docx")
        except Exception as e:
            log.append(f"! Word 报告生成失败：{e}")
        # 可独立运行的复现脚本
        script_py = generate_script(result, source_file, target_col, task_type)
        (out_dir / "script.py").write_text(script_py, encoding="utf-8")
        # 结果 JSON（供 /api/result/{task_id} 查询）
        (out_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        # 训练日志
        (out_dir / "training.log").write_text(
            "\n".join(log.lines), encoding="utf-8")

    _progress("done", 100, f"全部完成，总耗时 {result['training_time_total_seconds']}s")
    return result
