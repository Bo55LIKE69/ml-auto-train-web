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


def evaluate_classification(y_true, y_pred, y_proba=None):
    """
    计算分类指标与混淆矩阵。
    y_proba 传入 predict_proba 结果时，额外计算二分类 AUC。
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
        "auc": None,
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["auc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
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
