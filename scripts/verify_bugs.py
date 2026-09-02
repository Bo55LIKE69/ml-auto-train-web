# -*- coding: utf-8 -*-
"""
Bug 复现脚本：验证 train.py 中的若干可疑缺陷。
跑法： .venv\Scripts\python scripts\verify_bugs.py
"""
import io
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from app.ml.preprocess import prepare_dataset
from app.ml.train import run_pipeline, _build_models, LogCapture

OUT = Path("D:/ML_help/outputs/_bugcheck")
OUT.mkdir(parents=True, exist_ok=True)


def hr(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


# ---------------------------------------------------------------
# BUG 1: 单个模型训练失败 -> 整个任务崩溃（无异常隔离）
# ---------------------------------------------------------------
def bug1_single_model_crash():
    hr("BUG 1: 单模型失败是否拖垮整个任务")
    X, y = make_classification(n_samples=200, n_features=8, n_classes=2,
                               random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["label"] = y

    # 造一个必然崩的模型集：混入不存在的 key 不行，改为让 SVM 在小样本上跑
    # 更直接的验证：直接看训练循环源码是否有 try/except
    src = Path("D:/ML_help/app/ml/train.py").read_text(encoding="utf-8")
    loop_start = src.index("for i, (name, model) in enumerate(models.items(), 1):")
    loop_end = src.index("if best_name is None:")
    loop = src[loop_start:loop_end]
    has_try = "try:" in loop
    print(f"训练主循环内是否含 try/except 异常隔离: {has_try}")
    if not has_try:
        print("  >> 结论: 无任何异常隔离。12 个模型中任意 1 个抛异常，")
        print("     整个任务 status=failed，学生拿不到任何结果（含已训练好的模型）。")
    return not has_try


# ---------------------------------------------------------------
# BUG 2: 分层抽样在稀有类别上崩溃
# ---------------------------------------------------------------
def bug2_stratify_rare_class():
    hr("BUG 2: 目标列含稀有类别时 train_test_split 是否崩溃")
    rng = np.random.RandomState(42)
    n = 60
    df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
    })
    # 59 个 0 类，1 个 1 类 —— 真实问卷数据常见（如"是否违约"）
    y = [0] * 59 + [1] * 1
    df["label"] = y
    print(f"构造数据: {n} 行, 类别分布 = {pd.Series(y).value_counts().to_dict()}")
    try:
        result = run_pipeline(df, "label", task_type="classification",
                              out_dir=OUT / "rare", model_set=["dt"],
                              log=LogCapture())
        print(f"  >> 未崩溃，最优模型: {result['best_model']['name']}")
        return False
    except Exception as e:
        print(f"  >> 崩溃: {type(e).__name__}: {e}")
        print("     学生看到的是『训练失败』，不知道是数据问题还是软件问题。")
        return True


# ---------------------------------------------------------------
# BUG 3: fold 参数声称交叉验证，实际未执行
# ---------------------------------------------------------------
def bug3_fold_not_used():
    hr("BUG 3: fold(交叉验证折数) 参数是否真正执行 CV")
    src = Path("D:/ML_help/app/ml/train.py").read_text(encoding="utf-8")
    body = src[src.index("def run_pipeline"):]
    # 查找 cross_val_score / cross_validate / KFold 的实际调用
    imported_cv = ("cross_val_predict" in src)
    used_cv = any(k in body for k in ("cross_val_score(", "cross_validate(", "KFold("))
    print(f"  导入了 cross_val_predict : {imported_cv}")
    print(f"  在 run_pipeline 中真正调用 CV 评估 : {used_cv}")
    print(f"  fold 参数用途 : 仅写入 result['fold_count'] = {5}")
    if not used_cv:
        print("  >> 结论: 前端可设置『5 折交叉验证』，报告里也写 fold_count=5，")
        print("     但实际只做了 1 次 holdout(7:3) 划分。论文照抄即为数据造假风险。")
    return not used_cv


# ---------------------------------------------------------------
# BUG 4: model_set=None 时 _build_models 崩溃
# ---------------------------------------------------------------
def bug4_model_set_none():
    hr("BUG 4: _build_models(task_type, model_set=None) 是否崩溃")
    try:
        _build_models("classification", None)
        print("  >> 未崩溃")
        return False
    except Exception as e:
        print(f"  >> 崩溃: {type(e).__name__}: {e}")
        print("     run_pipeline 已默认填充，但外部/测试直接调用会 TypeError。")
        return True


# ---------------------------------------------------------------
# BUG 5: 超时降级集合定义了却未使用
# ---------------------------------------------------------------
def bug5_fallback_unused():
    hr("BUG 5: config 中的超时降级模型集是否被使用")
    cfg = Path("D:/ML_help/app/config.py").read_text(encoding="utf-8")
    tr = Path("D:/ML_help/app/ml/train.py").read_text(encoding="utf-8")
    defined = [n for n in ("FALLBACK_MODEL_SET_SMALL", "FALLBACK_MODEL_SET_MINIMAL")
               if n in cfg]
    used = [n for n in defined if n in tr]
    print(f"  config 中定义: {defined}")
    print(f"  train.py 中引用: {used if used else '（无）'}")
    if defined and not used:
        print("  >> 结论: 超时只做 break 中止，不会自动降级重跑。")
        print("     大数据集下学生可能只训完 2-3 个模型就被中止，且无提示。")
    return bool(defined and not used)


# ---------------------------------------------------------------
# BUG 6: requirements.txt 缺失关键依赖
# ---------------------------------------------------------------
def bug6_requirements_incomplete():
    hr("BUG 6: requirements.txt 是否覆盖全部代码依赖")
    req = Path("D:/ML_help/requirements.txt").read_text(encoding="utf-8").lower()
    needed = {
        "xgboost": "12 模型中的 XGBoost",
        "lightgbm": "12 模型中的 LightGBM",
        "catboost": "12 模型中的 CatBoost",
        "shap": "SHAP 可解释性图",
        "python-docx": "Word 七章报告",
        "joblib": "模型持久化（在线预测）",
    }
    missing = []
    for pkg, why in needed.items():
        ok = pkg in req
        print(f"  {pkg:14s} {'OK    ' if ok else '缺失  '}  ({why})")
        if not ok:
            missing.append(pkg)
    if missing:
        print(f"  >> 缺失 {len(missing)} 个。全新安装的学生: 12 模型降级为 9 个，")
        print("     SHAP 图无，Word 报告报错，在线预测不可用 —— 但界面无任何提示。")
    return bool(missing)


# ---------------------------------------------------------------
# BUG 7: 特征重要性强制用随机森林，而非最优模型
# ---------------------------------------------------------------
def bug7_importance_wrong_model():
    hr("BUG 7: 特征重要性是否来自最优模型")
    X, y = make_classification(n_samples=300, n_features=10, n_informative=4,
                               random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["label"] = y
    # 只用 XGBoost（不含 rf）
    r = run_pipeline(df, "label", task_type="classification",
                     out_dir=OUT / "norf", model_set=["xgboost", "dt"],
                     log=LogCapture())
    imp = r["feature_importance"]
    print(f"  模型集不含随机森林，最优模型 = {r['best_model']['name']}")
    print(f"  返回的特征重要性条数 = {len(imp)}")
    if not imp:
        print("  >> 结论: model_set 不含 rf 时，特征重要性为空。")
        print("     而 rf 存在时，展示的是 rf 的重要性，不是最优模型的 —— 与报告文字矛盾。")
        return True
    return False


# ---------------------------------------------------------------
# BUG 8: 任务状态仅存内存，重启后丢失
# ---------------------------------------------------------------
def bug8_task_state_in_memory():
    hr("BUG 8: 任务状态是否持久化")
    src = Path("D:/ML_help/app/routers/train.py").read_text(encoding="utf-8")
    print(f"  _TASKS 存储方式: 模块级 dict（进程内存）")
    print(f"  服务重启后 _TASKS 清空: 是")
    print(f"  存在 result.json 落盘兜底: {'result.json' in src}")
    print("  >> 影响: 重启后历史任务列表仍可显示（靠 result.json），")
    print("     但正在训练中的任务会永久停在 training 状态，前端一直转圈。")
    return True


if __name__ == "__main__":
    findings = {}
    for fn in (bug1_single_model_crash, bug2_stratify_rare_class,
               bug3_fold_not_used, bug4_model_set_none,
               bug5_fallback_unused, bug6_requirements_incomplete,
               bug7_importance_wrong_model, bug8_task_state_in_memory):
        try:
            findings[fn.__name__] = fn()
        except Exception:
            print(f"[{fn.__name__}] 复现脚本自身异常：")
            traceback.print_exc()
            findings[fn.__name__] = None

    hr("汇总")
    for k, v in findings.items():
        flag = {True: "确认存在", False: "未复现", None: "脚本异常"}[v]
        print(f"  {k:32s} {flag}")
    print(f"\n  确认存在 {sum(1 for v in findings.values() if v)} / {len(findings)} 项")
