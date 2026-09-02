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

import joblib
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

from app.config import (CV_ENABLED, CV_MAX_SAMPLES, CV_MIN_SAMPLES,
                        DEFAULT_FOLD, DEFAULT_MODEL_SET,
                        FALLBACK_MODEL_SET_MINIMAL, FALLBACK_MODEL_SET_SMALL,
                        RANDOM_STATE, SVM_LINEAR_THRESHOLD,
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


def model_status(task_type: str, model_set=None) -> dict:
    """
    返回模型可用性清单：区分「已构建」与「因依赖缺失而不可用」的模型。
    供 /api/deps 运行环境自检使用，让界面能明确告诉学生
    "XGBoost 不可用：未安装 xgboost 库"，而不是让模型凭空消失。
    """
    model_set = _resolve_model_set(model_set)
    catalog = MODEL_CATALOG.get(task_type, {})
    built = (_build_models(task_type, model_set)
             if task_type == "classification" else _build_regressors(model_set))
    built_names = set(built.keys())

    available, unavailable = [], []
    for key, meta in catalog.items():
        if key not in model_set:
            continue          # 未选入本次模型集，不算不可用
        if meta["name"] in built_names:
            available.append({"key": key, "name": meta["name"],
                              "desc": meta.get("desc", "")})
        else:
            unavailable.append({
                "key": key, "name": meta["name"],
                "pip_name": key if key in ("xgboost", "lightgbm", "catboost") else None,
                "reason": f"未安装 {key} 库，执行 pip install {key} 后可用",
            })
    return {"available": available, "unavailable": unavailable,
            "total": len(available) + len(unavailable),
            "ready": len(available)}


def _resolve_model_set(model_set) -> list:
    """
    规范化模型集参数：None / 空 / 非法类型统一回退为 DEFAULT_MODEL_SET。
    修复：直接传 None 时 `"lr" in None` 会抛 TypeError。
    """
    if not model_set:
        return list(DEFAULT_MODEL_SET)
    if isinstance(model_set, str):
        return [model_set]
    return list(model_set)


def _svm_kernel(n_samples=None) -> str:
    """
    SVM 核函数选择：RBF 在小样本上更准，但复杂度约 O(n²)~O(n³)；
    样本数超过阈值时切线性核，避免学生等几十分钟以为卡死。
    """
    if n_samples is not None and n_samples > SVM_LINEAR_THRESHOLD:
        return "linear"
    return "rbf"


def _build_models(task_type: str, model_set: list = None, n_samples=None) -> dict:
    """
    按规格书附录 A 构建模型字典。
    model_set 为缩写列表（如 ["lr","rf","xgboost"]），None 表示全部可用。
    n_samples 用于 SVM 核函数选择（大数据自动切线性核）。
    """
    model_set = _resolve_model_set(model_set)
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
        clf["SVM"] = SVC(kernel=_svm_kernel(n_samples), probability=True,
                         random_state=rs)
    return clf


def _build_regressors(model_set: list = None, n_samples=None) -> dict:
    """回归模型集（规格书附录 A 回归部分）。"""
    model_set = _resolve_model_set(model_set)
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


def _safe_stratify(y, warnings: list):
    """
    分层抽样安全性检查。
    当目标列存在样本数 < 2 的稀有类别时，sklearn 的 stratify 会直接抛
    ValueError（"least populated class has only 1 member"），导致整个任务失败。
    此处改为：能分层就分层，不能分层则降级为随机划分并明确告知用户。
    返回 (stratify 参数, 提示信息|None)
    """
    _, counts = np.unique(y, return_counts=True)
    if counts.size == 0:
        return None, "目标列无有效取值，无法划分"
    if counts.min() >= 2:
        return y, None
    n_rare = int((counts < 2).sum())
    msg = (f"目标列有 {n_rare} 个类别的样本数不足 2 条，无法按类别分层抽样，"
           f"已自动改用随机划分（建议补充稀有类别样本，或将稀有类别合并为『其他』）")
    if warnings is not None:
        warnings.append("! " + msg)
    return None, msg


def _validate_classification_target(y, warnings: list):
    """
    分类任务目标列合法性校验：只有 1 个类别时无法训练，给出可读的错误提示。
    同时提示极端不平衡（少数类占比 < 5%）。
    """
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError(
            "目标列只有 1 个取值，无法进行分类任务。"
            "请换一个目标列，或将任务类型改为『回归』"
            + ("（当前唯一取值：" + str(classes[0]) + "）" if classes.size else "")
        )
    if classes.size > 1:
        minority_ratio = counts.min() / counts.sum()
        if minority_ratio < 0.05:
            warnings.append(
                f"! 类别严重不平衡：最少类别仅占 {minority_ratio:.1%}"
                f"（{counts.min()}/{counts.sum()} 条），"
                f"模型可能偏向多数类，建议过采样或调整类别权重"
            )


def _resolve_cv_fold(task_type, y, fold, n_samples, warnings: list):
    """
    确定交叉验证的实际折数与是否启用。
    护栏：
      1) 显式 fold < 2 视为不启用
      2) 样本数 > CV_MAX_SAMPLES 自动跳过（大样本上 K 折代价过高）
      3) 分类任务折数不能超过最少类别的样本数（StratifiedKFold 硬约束）
      4) 折数不能超过样本数
    返回 (实际折数|None, 跳过原因|None)
    """
    if not fold or int(fold) < 2:
        return None, "未启用（fold < 2）"
    fold = int(fold)
    if n_samples > CV_MAX_SAMPLES:
        return None, f"样本数 {n_samples} 超过 CV 上限 {CV_MAX_SAMPLES}，已自动跳过"
    if n_samples < max(CV_MIN_SAMPLES, fold * 2):
        return None, f"样本数 {n_samples} 过少，K 折方差过大，已自动跳过"

    if task_type == "classification":
        _, counts = np.unique(y, return_counts=True)
        if counts.size and counts.min() < 2:
            return None, "存在样本数不足 2 的稀有类别，无法做分层 K 折，已跳过"
        # StratifiedKFold：每折每个类别至少 1 条 → 折数不能超过最少类别样本数
        fold = min(fold, int(counts.min()))
    fold = min(fold, n_samples)
    if fold < 2:
        return None, "折数被压缩至 1，无法交叉验证"
    return fold, None


def _cross_validate(task_type, model, preprocessor, X, y, fold, random_state):
    """
    对单个模型做 K 折交叉验证，返回 (均值, 标准差)。
    关键：用 Pipeline(clone(preprocessor), clone(model)) 包装，
    使每一折内部独立拟合预处理，彻底避免数据泄漏（答辩必问点）。
    任何异常都向上传播，由调用方决定是否计入结果。
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline as SkPipeline

    if task_type == "classification":
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
        scoring = "f1_macro"
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=fold, shuffle=True, random_state=random_state)
        scoring = "r2"

    pipe = SkPipeline([("preprocessor", clone(preprocessor)), ("model", clone(model))])
    scores = np.asarray(cross_val_score(pipe, X, y, cv=cv, scoring=scoring,
                                        n_jobs=-1), dtype=float)
    # 某些折可能因类别缺失/指标无定义而返回 NaN，必须剔除后再聚合，
    # 否则 NaN 会污染均值并让模型选优静默失效（NaN 比较恒为 False）
    scores = scores[~np.isnan(scores)]
    if scores.size == 0:
        raise ValueError(f"{fold} 折交叉验证全部失败（评分无法计算）")
    return float(np.mean(scores)), float(np.std(scores))


def _extract_importance(model, feature_names, top=15):
    """
    从【任意】模型提取特征重要性，按以下优先级：
      1) 树/集成模型的 feature_importances_
      2) 线性模型的 |coef_|（多分类取各类绝对值均值）
      3) 排列重要性不适用（代价高），不支持则返回空
    返回 (重要性列表, 来源说明)
    """
    imp = None
    source = None
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        source = "feature_importances_"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1:
            imp = np.abs(coef)
        elif coef.ndim == 2:
            # 多分类：shape (n_classes, n_features) → 各类绝对值取均值
            imp = np.abs(coef).mean(axis=0)
        else:
            imp = np.abs(coef).reshape(coef.shape[0], -1).mean(axis=0)
        source = "coef_ (绝对值)"

    if imp is None or imp.size == 0:
        return [], None
    # 维度对齐保护：独热编码后特征名数量可能与系数长度不同
    n = min(len(imp), len(feature_names))
    if n == 0:
        return [], None
    imp = imp[:n]
    names = list(feature_names)[:n]
    order = np.argsort(imp)[::-1][:top]
    return (
        [{"feature": str(names[i]), "importance": round(float(imp[i]), 6)}
         for i in order],
        source,
    )


def _train_one(name, model, task_type, X_train_t, y_train, X_test_t, y_test):
    """
    训练并评估【单个】模型，异常不外抛。
    返回 (metrics, y_pred, error)：error 非 None 表示该模型失败。
    ★ 这是「单模型失败不拖垮全局」的关键：任何模型崩溃都被收敛为一条错误记录。
    """
    t0 = time.time()
    try:
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)
        if task_type == "classification":
            y_proba = (model.predict_proba(X_test_t)
                       if hasattr(model, "predict_proba") else None)
            metrics, _cm = evaluate_classification(y_test, y_pred, y_proba)
        else:
            metrics = evaluate_regression(y_test, y_pred)
        metrics["train_time_s"] = round(time.time() - t0, 2)
        return metrics, y_pred, None
    except Exception as e:
        msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        return None, None, msg


def _tune_best_model(task_type, model_name, model, X_train_t, y_train,
                     budget=20, random_state=42):
    """
    对最优模型做轻量 RandomizedSearchCV 调优（毕设『模型优化』章节）。
    仅在 best model 上做，控制迭代次数，避免阻塞。
    返回 (tuned_model, best_params, tuned_score, better: bool)。
    """
    from sklearn.model_selection import RandomizedSearchCV

    scoring = "f1_macro" if task_type == "classification" else "r2"
    if task_type == "classification":
        space = {
            "逻辑回归": {"C": [0.01, 0.1, 1, 10], "max_iter": [2000]},
            "K近邻": {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]},
            "决策树": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
            "随机森林": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
            "极端随机树": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
            "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0]},
            "梯度提升": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
            "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
            "LightGBM": {"n_estimators": [100, 200], "num_leaves": [31, 63]},
            "CatBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
            "SVM": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        }
    else:
        space = {
            "K近邻回归": {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]},
            "决策树回归": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
            "随机森林回归": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
            "极端随机树回归": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]},
            "AdaBoost回归": {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0]},
            "梯度提升回归": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
            "XGBoost回归": {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
            "LightGBM回归": {"n_estimators": [100, 200], "num_leaves": [31, 63]},
            "CatBoost回归": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        }
    params = space.get(model_name)
    if not params:
        return model, getattr(model, "get_params", lambda: {})(), None, False
    try:
        n_iter = min(budget, max(4, int(np.prod([len(v) for v in params.values()]) / 2)))
        search = RandomizedSearchCV(
            model, params, n_iter=max(4, n_iter), scoring=scoring,
            cv=3, n_jobs=-1, random_state=random_state, refit=True)
        search.fit(X_train_t, y_train)
        return (search.best_estimator_, search.best_params_,
                float(search.best_score_), True)
    except Exception:
        return model, getattr(model, "get_params", lambda: {})(), None, False


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
                 log=None, progress_cb=None, fe_opts=None, tune=False,
                 tune_budget=20, use_cv=None):
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
        fe_opts      特征工程选项 dict（缺失值策略/缩放/类别编码），None 用默认
        tune         是否对最优模型做超参调优（RandomizedSearchCV）
        tune_budget  RandomizedSearchCV 迭代次数上限
        use_cv       是否启用 K 折交叉验证，None 表示跟随 config.CV_ENABLED

    返回：结果 dict（可直接 JSON 序列化，含模型对比、最优模型、图 URL 等）
    """
    t_start = time.time()
    out_dir = Path(out_dir) if out_dir else None
    log = log or LogCapture()
    model_set = _resolve_model_set(model_set)
    if use_cv is None:
        use_cv = CV_ENABLED

    def _progress(stage, pct, msg):
        log.append(f"[{stage}] {msg}")
        if progress_cb:
            progress_cb(stage, pct, msg)

    # ---- 1. 预处理 ----
    X, y, _, preprocessor, meta = prepare_dataset(df, target_col, id_cols, fe_opts)
    if task_type not in ("auto", "classification", "regression"):
        raise ValueError(f"非法的 task_type: {task_type}")
    if task_type != "auto":
        meta["task_type"] = task_type          # 用户强制指定任务类型
    task_type = meta["task_type"]
    fe = meta.get("fe_opts", {})
    _progress("preprocess", 5,
              f"预处理完成：{meta['n_samples']} 样本 x {meta['n_features_raw']} 特征，任务类型={task_type}"
              + (f"，特征工程[缺失值={fe.get('impute_strategy')} 缩放={fe.get('scaler')} 编码={fe.get('cat_encoding')}]" if fe else ""))

    # ---- 2. 目标列编码 ----
    y_enc, y_encoder, class_names = _encode_y(task_type, y)
    if task_type == "regression":
        mask = ~np.isnan(y_enc)
        if not mask.all():
            meta["warnings"].append(f"目标列有 {int((~mask).sum())} 行无法转为数值，已剔除")
            X, y_enc = X[mask], y_enc[mask]
        if len(np.unique(y_enc)) < 2:
            raise ValueError("目标列有效取值不足 2 个，无法进行回归任务")
    else:
        # 分类任务：单类别 / 极端不平衡提前拦下，给出人话提示
        _validate_classification_target(y_enc, meta["warnings"])
    _progress("encode", 10, "目标列编码完成")

    # ---- 3. 数据划分（分类按标签分层抽样，稀有类别自动降级为随机划分）----
    stratify = None
    if task_type == "classification":
        stratify, strat_note = _safe_stratify(y_enc, meta["warnings"])
        if strat_note:
            _progress("split", 12, strat_note)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=stratify)

    # ---- 4. 预处理管道 fit（只 fit 训练集，防数据泄漏）----
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = [str(n) for n in preprocessor.get_feature_names_out()]
    _progress("split", 15, f"划分完成：训练集 {X_train_t.shape[0]}，测试集 {X_test_t.shape[0]}，预处理后 {len(feature_names)} 维")

    # ---- 4.5 交叉验证可行性判定（护栏：大样本/稀有类别自动跳过）----
    cv_fold, cv_skip_reason = None, "未启用"
    if use_cv:
        cv_fold, cv_skip_reason = _resolve_cv_fold(
            task_type, y_enc, fold, len(X), meta["warnings"])
    if cv_fold:
        _progress("cv", 16, f"启用 {cv_fold} 折交叉验证：每个模型额外评估 {cv_fold} 次")
    else:
        _progress("cv", 16, f"交叉验证未启用 — {cv_skip_reason}")

    # ---- 5. 批量训练（逐模型异常隔离 + 超时降级）----
    n_samples = len(X)
    models = (_build_models(task_type, model_set, n_samples=n_samples)
              if task_type == "classification"
              else _build_regressors(model_set, n_samples=n_samples))
    if not models:
        raise ValueError("没有可用的模型（所选模型集为空或依赖未安装）")

    results = []
    failures = []
    model_refs, preds = {}, {}
    best_name, best_score = None, -float("inf")
    total = len(models)
    timed_out = False
    _score_key = "F1" if task_type == "classification" else "R2"

    def _run_one(name, model, idx_label=""):
        """训练+评估+CV 单个模型并登记结果，返回是否成功。"""
        nonlocal best_name, best_score
        metrics, y_pred, err = _train_one(
            name, model, task_type, X_train_t, y_train, X_test_t, y_test)
        if err is not None:
            failures.append({"name": name, "error": err})
            log.append(f"    {name:14s} -> 失败（已跳过，不影响其他模型）：{err}")
            return False

        # K 折交叉验证（失败不影响 holdout 主指标）
        if cv_fold:
            try:
                cv_mean, cv_std = _cross_validate(
                    task_type, model, preprocessor, X, y_enc, cv_fold, random_state)
                metrics["cv_mean"] = round(cv_mean, 4)
                metrics["cv_std"] = round(cv_std, 4)
                log.append(f"    {name:14s} -> {cv_fold}折CV {cv_mean:.4f} ± {cv_std:.4f}")
            except Exception as e:
                log.append(f"    {name:14s} -> CV 失败（已跳过）：{e}")

        model_refs[name], preds[name] = model, y_pred
        results.append({"name": name, "metrics": metrics})
        score = metrics.get("f1", metrics.get("r2")) or 0.0
        log.append(f"    {name:14s} -> {_score_key}={score} 耗时 {metrics['train_time_s']}s"
                   + idx_label)
        # CV 启用时以 K 折均值为选优依据：单次 holdout 划分偶然性大，
        # K 折均值更能反映模型真实泛化能力（毕设方法章节的标准写法）
        rank_score = metrics["cv_mean"] if (cv_fold and "cv_mean" in metrics) else score
        if rank_score > best_score:
            best_name, best_score = name, rank_score
        return True

    for i, (name, model) in enumerate(models.items(), 1):
        if time.time() - t_start > timeout:
            timed_out = True
            log.append(f"! 训练超时（>{timeout}s），中止剩余模型")
            break
        _progress("train", 16 + int(58 * (i - 1) / max(total, 1)),
                  f"训练模型 {i}/{total}：{name}")
        _run_one(name, model, idx_label=f"（{i}/{total}）")

    # ---- 5.2 超时降级：成功模型过少时，用精简模型集补跑 ----
    # 注意：降级阶段使用【独立的时间预算】。若沿用 t_start，预算早已耗尽，
    # 补跑会在第一个模型处立刻放弃，降级形同虚设。
    if timed_out and len(results) < 3:
        fb_deadline = time.time() + timeout
        log.append(f"! 超时但仅完成 {len(results)}/{total} 个模型，启动降级补跑"
                   f"（额外预算 {timeout}s）")
        for fallback_set in (FALLBACK_MODEL_SET_SMALL, FALLBACK_MODEL_SET_MINIMAL):
            todo = [m for m in fallback_set if m not in model_refs]
            if not todo:
                break
            fb_models = (_build_models(task_type, todo, n_samples=n_samples)
                         if task_type == "classification"
                         else _build_regressors(todo, n_samples=n_samples))
            # ★ 必须用【中文模型名】去重：todo 是英文缩写（如 "lr"），
            #   而 model_refs 的键是中文名（如 "逻辑回归"），按缩写过滤永远判断
            #   为"未训练"，会把已训完的模型再训一遍，对比表出现重复行。
            fb_models = {n: m for n, m in fb_models.items() if n not in model_refs}
            if not fb_models:
                continue
            _progress("fallback", 76, f"超时降级：补跑精简模型集 {list(fb_models)}")
            for name, model in fb_models.items():
                if time.time() > fb_deadline:
                    log.append("! 降级预算耗尽，停止补跑")
                    break
                _run_one(name, model, idx_label="（降级补跑）")
            if len(results) >= 3:
                break

    if best_name is None:
        detail = "；".join(f"{f['name']}({f['error']})" for f in failures[:3])
        raise RuntimeError(
            f"所有 {total} 个模型均训练失败，请检查数据或模型选择。失败原因：{detail}")

    # 对比表按选优口径降序排列，便于直接抄进论文
    if cv_fold:
        results.sort(key=lambda r: r["metrics"].get("cv_mean", -9e9), reverse=True)
    else:
        results.sort(key=lambda r: r["metrics"].get(
            "f1", r["metrics"].get("r2", -9e9)) or -9e9, reverse=True)

    best_metrics = next(r["metrics"] for r in results if r["name"] == best_name)
    best_pred = preds[best_name]
    best_model_ref = model_refs[best_name]
    _progress("train", 80, f"最优模型：{best_name}（{best_score}）"
              + (f"，{len(failures)} 个模型失败已跳过" if failures else ""))

    # ---- 5.5 超参调优（可选，仅对最优模型做轻量 RandomizedSearchCV）----
    tuned_info = {"enabled": False}
    if tune:
        _progress("tune", 82, f"对最优模型 {best_name} 启动超参调优（预算 {tune_budget} 次）...")
        try:
            tuned_model, tuned_params, tuned_cv, improved = _tune_best_model(
                task_type, best_name, best_model_ref, X_train_t, y_train,
                budget=tune_budget, random_state=random_state)
            if improved:
                best_model_ref = tuned_model
                pred_t = tuned_model.predict(X_test_t)
                if task_type == "classification":
                    proba_t = tuned_model.predict_proba(X_test_t) if hasattr(tuned_model, "predict_proba") else None
                    tm, _ = evaluate_classification(y_test, pred_t, proba_t)
                    base_score = best_metrics["f1"]
                    new_score = tm["f1"]
                else:
                    tm = evaluate_regression(y_test, pred_t)
                    base_score = best_metrics["r2"]
                    new_score = tm["r2"]
                best_pred = pred_t
                best_metrics = tm
                # 用调优后分数更新对比表中的最优项
                for r in results:
                    if r["name"] == best_name:
                        r["metrics"] = tm
                tuned_info = {
                    "enabled": True, "improved": new_score > base_score,
                    "base_score": round(float(base_score), 4),
                    "tuned_score": round(float(new_score), 4),
                    "best_params": tuned_params,
                }
                _progress("tune", 85, f"调优完成：{best_name} 分数 {base_score} -> {new_score}")
            else:
                tuned_info = {"enabled": True, "improved": False,
                              "base_score": best_score,
                              "tuned_score": best_score,
                              "best_params": tuned_params}
                _progress("tune", 85, f"调优未显著提升 {best_name}，沿用默认参数")
        except Exception as e:
            log.append(f"! 超参调优失败（沿用默认模型）：{e}")
            tuned_info = {"enabled": True, "improved": False, "error": str(e)}

    # ---- 6. 特征重要性：优先取【最优模型】自身，随机森林仅作兜底 ----
    # 修复点：旧实现强制用随机森林，导致 (a) 模型集不含 rf 时重要性为空，
    # (b) 展示的重要性与报告所述"最优模型"不是同一个模型。
    rf_fallback = model_refs.get("随机森林") or model_refs.get("随机森林回归")

    def _model_tag(m):
        if m is None:
            return None
        if m is best_model_ref:
            return best_name
        if m is rf_fallback:
            return "随机森林"
        return "模型"

    # 图与表保持同源：优先选用能出图的模型（具备 feature_importances_）
    imp_model = None
    for cand in (best_model_ref, rf_fallback):
        if cand is not None and hasattr(cand, "feature_importances_"):
            imp_model = cand
            break

    importance, importance_source = [], None
    if imp_model is not None:
        importance, src = _extract_importance(imp_model, feature_names)
        if importance:
            importance_source = f"{_model_tag(imp_model)}（{src}）"
    else:
        # 线性模型只有 coef_：仍可出表，但不出图（避免图与表口径不一致）
        importance, src = _extract_importance(best_model_ref, feature_names)
        if importance:
            importance_source = (f"{best_name}（{src}）；该模型无 "
                                 f"feature_importances_，未生成重要性图")
    if importance:
        log.append(f"[importance] 特征重要性来源：{importance_source}")
    else:
        log.append("! 最优模型不支持特征重要性（无 feature_importances_ 与 coef_），已跳过")

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
        if imp_model is not None:
            p = out_dir / "feature_importance.png"
            plot_feature_importance(
                imp_model, feature_names, p,
                title=f"特征重要性（{_model_tag(imp_model)}）")
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

        # 评估增强图：学习曲线 / ROC / 残差（毕设分析章节常用，异常跳过）
        from app.ml.plots import (plot_learning_curve, plot_roc_curve,
                                  plot_residual)
        # 学习曲线（用未调优的 base 模型更贴近『默认 vs 调优』对比；调优后也画一份）
        try:
            if plot_learning_curve(model_refs[best_name], X_train_t, y_train,
                                    X_test_t, y_test, out_dir / "learning_curve.png",
                                    cv=cv_fold or 5, task_type=task_type):
                plot_files["learning_curve"] = "learning_curve.png"
        except Exception as e:
            log.append(f"! 学习曲线生成失败（已跳过）：{e}")
        if task_type == "classification":
            try:
                proba_best = best_model_ref.predict_proba(X_test_t) if hasattr(best_model_ref, "predict_proba") else None
                if proba_best is not None and plot_roc_curve(y_test, proba_best, class_names,
                                                             out_dir / "roc_curve.png"):
                    plot_files["roc_curve"] = "roc_curve.png"
            except Exception as e:
                log.append(f"! ROC 曲线生成失败（已跳过）：{e}")
        else:
            try:
                if plot_residual(y_test, best_pred, out_dir / "residual.png"):
                    plot_files["residual"] = "residual.png"
            except Exception as e:
                log.append(f"! 残差图生成失败（已跳过）：{e}")

        # 模型持久化（在线推理用）：preprocessor + best model + 元信息
        try:
            artifacts = {
                "preprocessor": preprocessor,
                "model": best_model_ref,
                "model_name": best_name,
                "task_type": task_type,
                "class_names": class_names,
                "feature_names": feature_names,
                "target_col": target_col,
                "tuned": tuned_info.get("enabled", False),
                "random_state": random_state,
            }
            joblib.dump(artifacts, out_dir / "model_artifacts.joblib")
            plot_files["model_file"] = "model_artifacts.joblib"
            log.append("[model] 已持久化最优模型 model_artifacts.joblib（可在『在线预测』页推理新数据）")
        except Exception as e:
            log.append(f"! 模型持久化失败（已跳过）：{e}")

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
        "fe_opts": fe,
        "models": results,
        "best_model": {
            "name": best_name,
            "metrics": best_metrics,
            "reason": (
                f"{'F1(macro)' if task_type == 'classification' else 'R²'}"
                + (f" 的 {cv_fold} 折交叉验证均值最高" if cv_fold
                   else " 最高（测试集留出法）")
            ),
        },
        "tuned": tuned_info,
        "feature_importance": importance,
        "feature_importance_source": importance_source,
        # 交叉验证：记录实际生效折数与跳过原因，杜绝"声称做了 CV 其实没做"
        "cv": {
            "enabled": cv_fold is not None,
            "fold": cv_fold,
            "scoring": "f1_macro" if task_type == "classification" else "r2",
            "skip_reason": None if cv_fold else cv_skip_reason,
        },
        # 训练失败但被跳过、未影响整体的模型（旧实现会让整个任务失败）
        "failed_models": failures,
        "plots": {k: f"/api/download/{task_id}/{v}" for k, v in plot_files.items()},
        "report_url": f"/api/download/{task_id}/report.md",
        "docx_url": f"/api/download/{task_id}/report.docx",
        "script_url": f"/api/download/{task_id}/script.py",
        "model_url": f"/api/download/{task_id}/model_artifacts.joblib",
        "training_time_total_seconds": round(time.time() - t_start, 2),
        # fold_count 记录真实生效的折数（未启用 CV 时为 None），避免误导
        "fold_count": cv_fold,
        "session_id": random_state,
    }

    if out_dir is not None:
        # pipeline_ir 中间表示记账（规格书核心模块2）
        sort_metric = "F1" if task_type == "classification" else "R2"
        ir = new_ir(task_id, source_file, target_col, task_type, model_set,
                    sort_metric, fold, random_state)
        ir["setup_config"].update({
            "impute_strategy": fe.get("impute_strategy", "auto"),
            "scaler": fe.get("scaler", "auto"),
            "cat_encoding": fe.get("cat_encoding", "auto"),
        })
        ir = update_data_info(ir, df, meta["drop_cols"])
        best_params = {}
        if hasattr(best_model_ref, "get_params"):
            best_params = best_model_ref.get_params()
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
        script_py = generate_script(result, source_file, target_col, task_type, fe)
        (out_dir / "script.py").write_text(script_py, encoding="utf-8")
        # 结果 JSON（供 /api/result/{task_id} 查询）
        (out_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        # 训练日志
        (out_dir / "training.log").write_text(
            "\n".join(log.lines), encoding="utf-8")

    _progress("done", 100, f"全部完成，总耗时 {result['training_time_total_seconds']}s")
    return result
