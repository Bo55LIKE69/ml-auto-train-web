# -*- coding: utf-8 -*-
"""
可视化模块：混淆矩阵 / 真实-预测散点图 / 特征重要性图。
所有图片保存到 outputs/<task_id>/ 目录，返回文件名供前端展示。
"""
import matplotlib

# 强制使用非交互后端（Agg）：服务端在后台线程出图，GUI 后端（tkagg 等）
# 会在非主线程触发 Tcl 崩溃，导致整个 uvicorn 进程退出（已踩坑）。
matplotlib.use("Agg")

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
    """
    分类任务：混淆矩阵热力图。

    注意：必须显式传 labels=全部已知类别。若测试集恰好未覆盖某些类别
    （小样本 / 稀有类别场景极常见），from_predictions 只会输出实际出现的
    类别数维度，导致刻度数与 display_labels 数不匹配而抛
    "FixedLocator locations does not match the number of labels"，
    使整个训练任务在最后的绘图阶段失败。
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    labels = None
    if class_names:
        try:
            # 以全部类别编号固定矩阵维度，保证刻度与标签一一对应
            labels = list(range(len(class_names)))
            ConfusionMatrixDisplay.from_predictions(
                y_true, y_pred, labels=labels, display_labels=class_names,
                ax=ax, cmap="Blues")
        except Exception:
            ax.clear()
            labels = None
    if labels is None:
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
                            title="特征重要性"):
    """
    特征重要性条形图。
    模型由调用方传入（不再固定为随机森林），title 建议标注实际模型名，
    避免图注与报告文字不一致。
    """
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


def plot_metrics_comparison(results, task_type, save_path,
                            title="模型指标对比（测试集）"):
    """
    指标对比横向柱状图（规格书第四章：指标柱状对比图）。
    results: [{name, metrics}]，分类取 F1，回归取 R²。
    """
    names = [r["name"] for r in results]
    key = "f1" if task_type == "classification" else "r2"
    vals = [r["metrics"].get(key, 0) for r in results]
    colors = ["#2ecc71" if v == max(vals) else "#4C72B0" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(names))), dpi=120)
    ax.barh(range(len(names))[::-1], vals, color=colors)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names)
    ax.set_xlabel("F1" if key == "f1" else "R²")
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, len(names) - 1 - i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_correlation_heatmap(df, save_path, max_features=30,
                             title="特征相关性热力图"):
    """
    数值特征相关性热力图（数据探索增强）。
    仅保留数值列；特征数超过 max_features 时按方差取 Top N，避免图过密。
    df: 原始 DataFrame（含目标列）
    """
    import pandas as pd

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return False  # 数值列不足，跳过
    if num_df.shape[1] > max_features:
        vars_ = num_df.var().sort_values(ascending=False)
        num_df = num_df[vars_.index[:max_features]]

    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(max(7, corr.shape[1] * 0.55),
                                    max(6, corr.shape[0] * 0.55)), dpi=120)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


# ========== 评估增强图（毕设分析章节常用） ==========

def plot_learning_curve(model, X_train, y_train, X_test, y_test, save_path,
                        cv=5, task_type="classification",
                        title="学习曲线（训练/验证得分）"):
    """
    学习曲线：判断模型是否过拟合/欠拟合（导师常问点）。
    用 sklearn.learning_curve 在训练集上递增样本量，绘制训练/交叉验证得分均值±标准差。

    修复的两个隐患：
      - 旧实现固定 scoring="accuracy"，回归任务的 y 是连续值，会抛
        "continuous is not supported"，导致该图在回归任务下永远生成失败；
      - cv 固定为 5，当目标列存在样本数 < 5 的稀有类别时 StratifiedKFold
        无法划分（5×5=25 次拟合里失败 22 次），并在日志里刷屏。
    """
    from sklearn.base import clone
    from sklearn.model_selection import KFold, StratifiedKFold, learning_curve

    try:
        y_train = np.asarray(y_train)
        n = len(y_train)
        want = int(cv or 5)
        if task_type == "classification":
            _, counts = np.unique(y_train, return_counts=True)
            min_count = int(counts.min()) if counts.size else 0
            if min_count < 2:
                return False      # 稀有类别无法分层，该图无意义，直接放弃
            k = max(2, min(want, min_count, n))
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            scoring = "accuracy"
        else:
            k = max(2, min(want, n))
            splitter = KFold(n_splits=k, shuffle=True, random_state=42)
            scoring = "r2"

        sizes, train_s, val_s = learning_curve(
            clone(model), X_train, y_train, cv=splitter,
            train_sizes=np.linspace(0.1, 1.0, 5), scoring=scoring,
            n_jobs=-1)
        fig, ax = plt.subplots(figsize=(6.5, 4.6), dpi=120)
        ax.plot(sizes, train_s.mean(1), "o-", color="#4C72B0", label="训练集得分")
        ax.fill_between(sizes, train_s.mean(1) - train_s.std(1),
                        train_s.mean(1) + train_s.std(1), alpha=0.12, color="#4C72B0")
        ax.plot(sizes, val_s.mean(1), "o-", color="#2ecc71", label="交叉验证得分")
        ax.fill_between(sizes, val_s.mean(1) - val_s.std(1),
                        val_s.mean(1) + val_s.std(1), alpha=0.12, color="#2ecc71")
        ax.set_xlabel("训练样本数")
        ax.set_ylabel("准确率" if task_type == "classification" else "R²")
        # 回归的 R² 可能为负，固定 (0, 1.05) 会把曲线压成一条线
        if task_type == "classification":
            ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return True
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return False


def plot_roc_curve(y_test, y_proba, class_names, save_path,
                  task_type="classification", title="ROC 曲线（宏平均 AUC）"):
    """
    ROC 曲线：分类任务宏平均（OvR）+ 每类曲线。
    y_proba: predict_proba 结果 (n_samples, n_classes)。
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_test = np.asarray(y_test).astype(int)
    y_proba = np.asarray(y_proba)
    n_classes = y_proba.shape[1]
    if n_classes < 2:
        return False
    try:
        y_bin = np.eye(n_classes)[y_test] if y_test.max() < n_classes else label_binarize(y_test, classes=range(n_classes))
        fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=120)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_classes, 2)))
        # 宏平均 AUC：逐类计算后取均值（避免 ravel 长度不一致）
        macro_fpr = np.linspace(0, 1, 100)
        macro_tpr_interp = []
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            macro_tpr_interp.append(np.interp(macro_fpr, fpr_i, tpr_i))
            ax.plot(fpr_i, tpr_i, color=colors[i], lw=1.4,
                    label=f"{class_names[i] if i < len(class_names) else i} (AUC={auc(fpr_i, tpr_i):.3f})")
        mean_tpr = np.mean(macro_tpr_interp, axis=0)
        macro_auc = auc(macro_fpr, mean_tpr)
        ax.plot(macro_fpr, mean_tpr, "k--", lw=2, label=f"宏平均 (AUC={macro_auc:.3f})")
        ax.plot([0, 1], [0, 1], "r:", lw=1)
        ax.set_xlabel("假正率 (FPR)")
        ax.set_ylabel("真正率 (TPR)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return True
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return False


def plot_residual(y_true, y_pred, save_path, title="残差图（真实值 vs 预测值）"):
    """
    回归残差图：残差(真实-预测) vs 预测值，用于检测异方差/非线性。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4.6), dpi=120)
    ax.scatter(y_pred, resid, alpha=0.6, edgecolors="k", linewidths=0.5)
    ax.axhline(0, color="r", ls="--", lw=1.5)
    ax.set_xlabel("预测值")
    ax.set_ylabel("残差 (真实 - 预测)")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def plot_shap_summary(model, X_sample, feature_names, save_path,
                      title="SHAP 特征重要性（BeeSwarm）"):
    """
    SHAP 可解释性 BeeSwarm 图（最优模型）。
    依赖 shap 库；采样上限由调用方控制（默认 100 行）。
    回归模型使用 explainer(model.predict)；分类模型取类别 1 的 SHAP 值。
    """
    import shap

    try:
        # 树模型（XGBoost/LightGBM/CatBoost/随机森林等）优先用 TreeExplainer
        tree_like = hasattr(model, "get_booster") or hasattr(model, "n_estimators") or \
            hasattr(model, "feature_importances_")
        if tree_like:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample)
        # 分类模型 shap_values 是列表（每类一个），取最后一类（正类）
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * min(len(feature_names), 20))), dpi=120)
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          show=False, max_display=min(len(feature_names), 20))
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        # SHAP 对部分模型（如 SVC probability）不支持，降级返回 False
        try:
            plt.close("all")
        except Exception:
            pass
        return False
