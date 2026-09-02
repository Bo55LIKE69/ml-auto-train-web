# -*- coding: utf-8 -*-
"""
修复验证脚本：对每个已修缺陷做行为级回归测试（不扫描源码字符串）。
跑法： .venv\Scripts\python scripts\verify_fixes.py
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

import app.ml.train as T
from app.ml.train import run_pipeline, LogCapture

OUT = Path("D:/ML_help/outputs/_fixcheck")
OUT.mkdir(parents=True, exist_ok=True)

PASS, FAIL = [], []

# ---- 警告追踪：定位每个 sklearn/matplotlib 警告归属哪个测试段 ----
import warnings

CURRENT = ["(初始化)"]
WARNS = []
_old_showwarning = warnings.showwarning


def _hook(message, category, filename, lineno, *a, **k):
    WARNS.append((CURRENT[0], category.__name__, str(message)[:90],
                  Path(filename).name, lineno))


warnings.showwarning = _hook


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def hr(t):
    CURRENT[0] = t
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70)


def clf_df(n=300, nf=8):
    X, y = make_classification(n_samples=n, n_features=nf, n_informative=4,
                               random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(nf)])
    df["label"] = y
    return df


# ------------------------------------------------------------------
hr("修复 1：单个模型训练失败，不应拖垮整个任务")


class BoomModel:
    """模拟一个必然崩溃的模型（如依赖版本不兼容、数据不适配）。"""

    def fit(self, X, y):
        raise RuntimeError("模拟崩溃：该模型不支持当前数据")

    def predict(self, X):
        raise RuntimeError("boom")


_orig_build = T._build_models


def _patched_build(task_type, model_set=None, n_samples=None):
    d = _orig_build(task_type, model_set, n_samples)
    d["注入失败模型"] = BoomModel()   # 放在最后，验证它不会中断前面已成功的模型
    return d


T._build_models = _patched_build
try:
    r = run_pipeline(clf_df(), "label", task_type="classification",
                     out_dir=OUT / "fail_isolate",
                     model_set=["lr", "dt", "rf"], use_cv=False, log=LogCapture())
    names = [m["name"] for m in r["models"]]
    failed = [f["name"] for f in r["failed_models"]]
    check("任务整体成功完成", len(r["models"]) >= 3, f"成功 {names}")
    check("失败模型被单独记录而非中断任务", failed == ["注入失败模型"], f"失败={failed}")
    check("成功模型的指标完整", all("f1" in m["metrics"] for m in r["models"]))
except Exception as e:
    check("任务整体成功完成", False, f"仍崩溃: {type(e).__name__}: {e}")
finally:
    T._build_models = _orig_build

# 全部模型失败时，应给出可读错误而非裸异常
T._build_models = lambda *a, **k: {"崩1": BoomModel(), "崩2": BoomModel()}
try:
    run_pipeline(clf_df(60), "label", task_type="classification",
                 out_dir=OUT / "all_fail", use_cv=False, log=LogCapture())
    check("全部失败时抛出可读错误", False, "未抛异常")
except RuntimeError as e:
    check("全部失败时抛出可读错误", "失败原因" in str(e), str(e)[:70] + "...")
except Exception as e:
    check("全部失败时抛出可读错误", False, f"异常类型不当: {type(e).__name__}")
finally:
    T._build_models = _orig_build


# ------------------------------------------------------------------
hr("修复 2：目标列含稀有类别（样本数<2）不再崩溃")
n = 60
df = pd.DataFrame({"f1": np.random.RandomState(0).randn(n),
                   "f2": np.random.RandomState(1).randn(n)})
df["label"] = [0] * 59 + [1] * 1          # 59:1，问卷数据极常见
try:
    r = run_pipeline(df, "label", task_type="classification",
                     out_dir=OUT / "rare", model_set=["dt", "lr"],
                     use_cv=False, log=LogCapture())
    warned = any("稀有" in w or "分层" in w for w in r["warnings"])
    check("稀有类别数据可正常训练", len(r["models"]) >= 1, f"模型 {[m['name'] for m in r['models']]}")
    check("给出分层抽样降级的明确提示", warned,
          next((w for w in r["warnings"] if "分层" in w or "稀有" in w), "")[:60])
except Exception as e:
    check("稀有类别数据可正常训练", False, f"仍崩溃: {type(e).__name__}: {e}")

# 单类别目标列 —— 应给出人话提示
df1 = clf_df(50); df1["label"] = 1
try:
    run_pipeline(df1, "label", task_type="classification",
                 out_dir=OUT / "single", model_set=["dt"], use_cv=False,
                 log=LogCapture())
    check("单类别目标列给出可读提示", False, "未拦截")
except ValueError as e:
    check("单类别目标列给出可读提示", "只有 1 个取值" in str(e), str(e)[:60])


# ------------------------------------------------------------------
hr("修复 3：K 折交叉验证真的执行了")
r = run_pipeline(clf_df(200), "label", task_type="classification",
                 out_dir=OUT / "cv", model_set=["lr", "dt", "rf"],
                 fold=5, use_cv=True, log=LogCapture())
metrics0 = r["models"][0]["metrics"]
check("CV 已启用且折数为 5", r["cv"]["enabled"] and r["cv"]["fold"] == 5, str(r["cv"]))
check("每个模型都有 cv_mean / cv_std",
      all("cv_mean" in m["metrics"] and "cv_std" in m["metrics"] for m in r["models"]),
      f"示例 cv_mean={metrics0.get('cv_mean')} cv_std={metrics0.get('cv_std')}")
check("CV 均值与 holdout 分数接近（说明实现正确，非随机填充）",
      abs(metrics0["cv_mean"] - metrics0["f1"]) < 0.25,
      f"cv_mean={metrics0['cv_mean']} vs holdout f1={metrics0['f1']}")
check("选优依据为 CV 均值", "交叉验证" in r["best_model"]["reason"], r["best_model"]["reason"])
check("对比表按 CV 均值降序",
      all(r["models"][i]["metrics"]["cv_mean"] >= r["models"][i + 1]["metrics"]["cv_mean"]
          for i in range(len(r["models"]) - 1)),
      " → ".join(f"{m['name']}:{m['metrics']['cv_mean']}" for m in r["models"]))

# 大样本护栏
r2 = run_pipeline(clf_df(120), "label", task_type="classification",
                  out_dir=OUT / "cv_skip", model_set=["dt"], fold=5,
                  use_cv=True, log=LogCapture())
check("小样本/正常样本 CV 仍启用", r2["cv"]["enabled"] is True, str(r2["cv"]))


# ------------------------------------------------------------------
hr("修复 4：_build_models(model_set=None) 不再崩溃")
for ms in (None, [], "rf"):
    try:
        d = T._build_models("classification", ms)
        check(f"model_set={ms!r} 安全处理", isinstance(d, dict) and len(d) > 0,
              f"返回 {len(d)} 个模型")
    except Exception as e:
        check(f"model_set={ms!r} 安全处理", False, f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------
hr("修复 5：特征重要性来自最优模型，而非固定的随机森林")
# 5a. 模型集不含 rf 时，重要性不应为空
r = run_pipeline(clf_df(200), "label", task_type="classification",
                 out_dir=OUT / "imp_norf", model_set=["xgboost", "dt"],
                 use_cv=False, log=LogCapture())
check("模型集不含随机森林时仍能给出特征重要性",
      len(r["feature_importance"]) > 0,
      f"来源={r.get('feature_importance_source')}，条数={len(r['feature_importance'])}")
src = (r.get("feature_importance_source") or "")
check("重要性来源标注为最优模型",
      r["best_model"]["name"] in src, f"最优={r['best_model']['name']}，来源={src}")

# 5b. 线性模型（只有 coef_）也应能出重要性表
r = run_pipeline(clf_df(200), "label", task_type="classification",
                 out_dir=OUT / "imp_lr", model_set=["lr"],
                 use_cv=False, log=LogCapture())
check("线性模型（无 feature_importances_）也能给出重要性",
      len(r["feature_importance"]) > 0,
      f"来源={r.get('feature_importance_source')}")


# ------------------------------------------------------------------
hr("修复 6：回归任务全流程 + CV 正常")
X, y = make_regression(n_samples=200, n_features=6, noise=0.1, random_state=42)
rdf = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
rdf["price"] = y
try:
    r = run_pipeline(rdf, "price", task_type="regression",
                     out_dir=OUT / "reg", model_set=["lr", "dt", "rf"],
                     fold=5, use_cv=True, log=LogCapture())
    check("回归任务训练成功", len(r["models"]) == 3, f"{[m['name'] for m in r['models']]}")
    check("回归 CV 使用 R² 评分", r["cv"]["scoring"] == "r2", str(r["cv"]))
    check("回归指标含 r2/mae/rmse",
          all(k in r["models"][0]["metrics"] for k in ("r2", "mae", "rmse")))
    check("回归特征重要性非空", len(r["feature_importance"]) > 0,
          f"来源={r.get('feature_importance_source')}")
except Exception as e:
    check("回归任务训练成功", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()


# ------------------------------------------------------------------
hr("修复 7：极端类别不平衡给出提示")
df = clf_df(200)
df["label"] = [0] * 194 + [1] * 6          # 少数类占 3%
try:
    r = run_pipeline(df, "label", task_type="classification",
                     out_dir=OUT / "imbalance", model_set=["dt"],
                     use_cv=False, log=LogCapture())
    check("极端不平衡被识别并提示",
          any("不平衡" in w for w in r["warnings"]),
          next((w for w in r["warnings"] if "不平衡" in w), "无提示")[:70])
except Exception as e:
    check("极端不平衡被识别并提示", False, f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------
hr("警告溯源（按测试段归类）")
warnings.showwarning = _old_showwarning
if not WARNS:
    print("  无警告")
else:
    seen = {}
    for seg, cat, msg, fn, ln in WARNS:
        key = (seg, cat, msg, fn, ln)
        seen[key] = seen.get(key, 0) + 1
    for (seg, cat, msg, fn, ln), cnt in sorted(seen.items(), key=lambda x: -x[1]):
        print(f"  x{cnt}  [{seg[:26]}] {cat}: {msg}")
        print(f"         @ {fn}:{ln}")

hr("汇总")
print(f"  通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
for f in FAIL:
    print(f"    - 失败：{f}")
sys.exit(1 if FAIL else 0)
