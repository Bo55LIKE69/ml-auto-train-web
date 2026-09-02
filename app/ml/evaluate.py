# -*- coding: utf-8 -*-
"""
模型评估模块：分类与回归指标计算（纯 sklearn，独立可测）。

分类指标：准确率 accuracy / 精确率 precision(macro) / 召回率 recall(macro)
          / F1(macro) / AUC（仅二分类）/ Kappa / MCC（规格书附录 B）
回归指标：R² / MAE / MSE / RMSE
"""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, cohen_kappa_score,
                             f1_score, matthews_corrcoef,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)

# CV 评分函数（分类用 macro-F1，回归用 R²），与 run_pipeline 的选优口径一致
CV_SCORING = {"classification": "f1_macro", "regression": "r2"}


def _safe_auc(y_true, y_proba):
    """
    计算 AUC：二分类取正类概率，多分类用 one-vs-rest macro。
    任何异常（测试集缺类、概率矩阵列数不匹配）都返回 None，不抛错。
    """
    if y_proba is None:
        return None
    y_proba = np.asarray(y_proba)
    if y_proba.ndim != 2 or y_proba.shape[1] < 2:
        return None
    classes = np.unique(y_true)
    try:
        if len(classes) == 2:
            # 二分类：正类概率列。若概率列数与真实类别数不一致（测试集恰好只含一类），放弃
            if y_proba.shape[1] != 2:
                return None
            return round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
        if len(classes) > 2 and y_proba.shape[1] >= len(classes):
            return round(float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro")), 4)
    except Exception:
        return None
    return None


def evaluate_classification(y_true, y_pred, y_proba=None):
    """
    计算分类指标与混淆矩阵。
    y_proba 传入 predict_proba 结果时，额外计算 AUC（二分类/多分类均可）。
    返回 (metrics dict, confusion matrix list)
    """
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "auc": _safe_auc(y_true, y_proba),
    }
    cm = confusion_matrix(y_true, y_pred).tolist()
    return metrics, cm


def evaluate_regression(y_true, y_pred):
    """计算回归指标：R² / MAE / MSE / RMSE。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse": round(mse, 4),
        "rmse": round(float(np.sqrt(mse)), 4),
    }
