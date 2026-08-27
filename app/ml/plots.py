# -*- coding: utf-8 -*-
"""
可视化模块：混淆矩阵 / 真实-预测散点图 / 特征重要性图。
所有图片保存到 outputs/<task_id>/ 目录，返回文件名供前端展示。
"""
import matplotlib

# 中文字体：优先 Microsoft YaHei / SimHei（Windows 自带），
# 不存在时回退 DejaVu Sans（英文标签），保证脚本在 Linux 也能跑。
from matplotlib import font_manager

_available = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
    if _f in _available:
        matplotlib.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, class_names, save_path,
                          title="混淆矩阵（最优模型 · 测试集）"):
    """分类任务：混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, ax=ax, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_scatter(y_true, y_pred, save_path,
                 title="真实值 vs 预测值（最优模型 · 测试集）"):
    """回归任务：真实-预测散点图，附 y=x 理想线。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x 理想线")
    ax.set_xlabel("真实值")
    ax.set_ylabel("预测值")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_feature_importance(model, feature_names, save_path, top_n=15,
                            title="特征重要性（随机森林）"):
    """特征重要性条形图：固定使用随机森林模型的 feature_importances_。"""
    imp = np.asarray(model.feature_importances_)
    order = np.argsort(imp)[::-1][:top_n]
    names = [str(feature_names[i]) for i in order]
    vals = imp[order]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(names))), dpi=120)
    ax.barh(range(len(names))[::-1], vals, color="#4C72B0")
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names)
    ax.set_xlabel("重要性得分")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
