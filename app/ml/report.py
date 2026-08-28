# -*- coding: utf-8 -*-
"""
报告生成模块：根据训练结果 dict 生成
1) Markdown 实验报告（可直接复制进论文）
2) 可独立运行的 Python 复现脚本（学生本地一键复现）
"""
import datetime


def generate_report(result, df, y, meta) -> str:
    """生成 Markdown 实验报告。result 为 run_pipeline 的返回值。"""
    task = "分类" if result["task_type"] == "classification" else "回归"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    target = result["target_col"]
    warnings = "\n".join(f"- {w}" for w in result["warnings"]) or "- 无"

    # 模型对比表格（Markdown 表）
    lines = ["| 模型 | " + _metric_headers(result) + " |"]
    lines.append("|" + "---|" * (3 + len(result["models"][0]["metrics"])) + "|")
    for m in result["models"]:
        cells = " | ".join(str(v) for v in m["metrics"].values())
        lines.append(f"| {m['name']} | {cells} |")
    model_table = "\n".join(lines)

    # 特征重要性表
    imp_lines = ["| 排名 | 特征 | 重要性 |", "|---|---|---|"]
    for i, item in enumerate(result["feature_importance"], 1):
        imp_lines.append(f"| {i} | {item['feature']} | {item['importance']} |")
    imp_table = "\n".join(imp_lines) if result["feature_importance"] else "（无随机森林模型，未生成）"

    best = result["best_model"]
    metric_str = "，".join(f"{k}={v}" for k, v in best["metrics"].items() if v is not None)

    # 图片引用
    plots_md = []
    if "confusion_matrix" in result["plots"]:
        plots_md.append(f"![混淆矩阵]({result['plots']['confusion_matrix']})")
    if "scatter" in result["plots"]:
        plots_md.append(f"![真实-预测散点图]({result['plots']['scatter']})")
    if "feature_importance" in result["plots"]:
        plots_md.append(f"![特征重要性]({result['plots']['feature_importance']})")

    return f"""# 表格数据集机器学习实验报告

> 生成时间：{now} ｜ 任务类型：{task} ｜ 目标标签：`{target}`

## 1. 数据集说明

- **样本数量**：{result['n_samples']}
- **原始特征数**：{result['n_features_raw']}（预处理后 {result['n_features_after_prep']} 维）
- **目标列**：`{target}`
- **任务类型**：{task}（{ '类别数：' + str(len(result['class_names'])) + '，类别：' + ' / '.join(result['class_names']) if result['class_names'] else '连续数值' }）

## 2. 数据问题说明

{warnings}

## 3. 实验设置

- 训练集 / 测试集划分：**7:3**（随机种子 42，可复现）
- 预处理：数值列缺失值填充**中位数**并标准化；类别列缺失值填充**众数**并**独热编码**；
  预处理管道仅在训练集上拟合，测试集只做变换，**避免数据泄漏**
- 已剔除无意义列：{('、'.join(result['drop_cols']) if result['drop_cols'] else '无')}
- 模型均使用默认参数（基础模型对比，未调参）

## 4. 模型结果对比

{model_table}

## 5. 最优模型

**{best['name']}**（{best['reason']}）

测试集指标：{metric_str}

## 6. 可视化

{chr(10).join(plots_md) if plots_md else '（无）'}

## 7. 实验结论

在本数据集（{result['n_samples']} 条样本）上，{best['name']} 表现最优，
{best['reason']}。特征重要性分析表明，
{_top_features_sentence(result)}。

## 8. 局限性与后续改进

- 数据量有限（{result['n_samples']} 条），模型性能存在偶然性，建议扩充样本
- 仅使用基础模型默认参数，未进行网格搜索调优，后续可对最优模型做 **GridSearchCV** 超参数优化
- 类别列采用独热编码，类别数较多时特征维度膨胀，可尝试 **目标编码（Target Encoding）**
- 未处理类别不平衡问题，若样本分布不均，可尝试 **SMOTE** 过采样或调整类别权重
- 后续可引入交叉验证（K-Fold）进一步验证模型稳定性

## 9. 复现方式

下载配套 Python 脚本（script.py），安装依赖后直接运行即可复现本实验全部结果。
"""


def _metric_headers(result):
    """模型对比表头：模型 | 各指标名。"""
    return " | ".join(result["models"][0]["metrics"].keys())


def _top_features_sentence(result):
    if not result["feature_importance"]:
        return "当前未获得特征重要性信息"
    top = result["feature_importance"][:3]
    return "、".join(f"**{t['feature']}**（{t['importance']}）" for t in top) + " 为影响预测结果的主要特征"


def generate_script(result, source_file, target_col, task_type, fe_opts=None) -> str:
    """
    生成可独立运行的 Python 复现脚本。
    脚本逻辑与 app/ml/train.py 的 run_pipeline 完全一致，
    学生在本机安装依赖后即可一键复现全部结果。
    fe_opts：特征工程选项 dict（缺失值策略/缩放/类别编码），脚本将按实际配置生成管道。
    """
    task = "classification" if task_type == "classification" else "regression"
    source = source_file or "你的数据文件.csv"
    metric_names = list(result["models"][0]["metrics"].keys())
    models_cfg = (
        """
# 分类候选：逻辑回归 / 随机森林 / 决策树 / SVM
MODELS = {
    "逻辑回归": LogisticRegression(max_iter=2000, random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "决策树": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}"""
        if task == "classification"
        else """
# 回归候选：线性回归 / 随机森林回归 / 决策树回归
MODELS = {
    "线性回归": LinearRegression(),
    "随机森林回归": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "决策树回归": DecisionTreeRegressor(random_state=42),
}"""
    )
    import_block = (
        """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             r2_score, mean_absolute_error, mean_squared_error)
import numpy as np"""
        if task == "classification"
        else """
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np"""
    )
    plot_block = (
        """
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
matplotlib 中文字体设置
"""
        if False
        else """import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import font_manager
_avail = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
    if _f in _avail:
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False"""
    )

    # 特征工程选项脚本片段（自动反映训练时实际配置，auto 为默认）
    fe_opts = fe_opts or {}
    _fe = {
        "impute_strategy": fe_opts.get("impute_strategy", "auto"),
        "scaler": fe_opts.get("scaler", "auto"),
        "cat_encoding": fe_opts.get("cat_encoding", "auto"),
    }
    fe_script = "{" + ", ".join(f'"{k}": "{v}"' for k, v in _fe.items()) + "}"
    # 按需补充 import（MinMaxScaler / LabelEncoder）
    if _fe["scaler"] == "minmax":
        import_block += "\nfrom sklearn.preprocessing import MinMaxScaler"
    if _fe["cat_encoding"] == "label":
        import_block += "\nfrom sklearn.preprocessing import OrdinalEncoder"

    return f'''# -*- coding: utf-8 -*-
"""
表格数据集机器学习自动训练 —— 结果复现脚本
================================================================
由「表格ML自动训练工具」自动生成（{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}）

用法：
    1. 安装依赖：  pip install pandas scikit-learn matplotlib
    2. 修改下方 DATA_PATH 与 TARGET_COL
    3. 运行：      python script.py
    4. 运行结束后，当前目录会生成：
       - confusion_matrix.png / scatter.png（最优模型可视化）
       - feature_importance.png（特征重要性）
       - report.md（实验报告）
"""
{import_block}
{plot_block}

# ==================== 可修改配置 ====================
DATA_PATH = r"{source}"    # 你的数据文件路径
TARGET_COL = "{target_col}"      # 目标标签列
TASK_TYPE = "{task}"          # classification / regression（自动判断失败时手动指定）
ID_COLS = []                  # 需要剔除的 ID 列，如 ["学号", "编号"]
TEST_SIZE = 0.3               # 测试集比例
RANDOM_STATE = 42             # 随机种子（保证可复现）
# ==================================================

{models_cfg}


def load_data(path):
    """读取 CSV/Excel（自动尝试常见编码）。"""
    if str(path).lower().endswith(".csv"):
        for enc in ("utf-8", "gbk", "gb18030", "utf-8-sig"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError("CSV 编码无法识别")
    return pd.read_excel(path)


def main():
    df = load_data(DATA_PATH)
    print(f"[1/7] 数据读取完成：{{df.shape[0]}} 行 x {{df.shape[1]}} 列")

    # ---- 预处理 ----
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL])
    mask = y.notna()
    X, y = X[mask], y[mask]

    # 剔除 ID 列
    for c in ID_COLS:
        if c in X.columns:
            X = X.drop(columns=[c])

    # 任务类型判定
    if TASK_TYPE == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_names = [str(c) for c in le.classes_]
        stratify = y
    else:
        y = pd.to_numeric(y, errors="coerce").astype(float).values
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]
        class_names = None
        stratify = None

    # 特征工程选项（与训练时一致，auto 为默认）
    FE_OPTS = {fe_script}

    # 预处理管道：按 FE_OPTS 构建
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c].dropna())]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c].dropna())]
    impute_strategy = FE_OPTS["impute_strategy"]
    scaler = FE_OPTS["scaler"]
    cat_encoding = FE_OPTS["cat_encoding"]
    num_impute = "median" if impute_strategy == "auto" else impute_strategy
    cat_impute = "most_frequent" if impute_strategy == "auto" else impute_strategy
    transformers = []
    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy=num_impute))]
        if scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif scaler != "none":
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        cat_steps = [("imputer", SimpleImputer(strategy=cat_impute))]
        if cat_encoding == "label":
            cat_steps.append(("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1)))
        else:
            cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore")))
        transformers.append(("cat", Pipeline(cat_steps), cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop")

    print(f"[2/7] 预处理完成：任务类型={{TASK_TYPE}}，特征工程={{FE_OPTS}}")

    # ---- 7:3 划分 ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify)
    pre.fit(X_train)
    X_train_t, X_test_t = pre.transform(X_train), pre.transform(X_test)
    print(f"[3/7] 划分完成：训练集 {{X_train_t.shape[0]}}，测试集 {{X_test_t.shape[0]}}")

    # ---- 批量训练与评估 ----
    results = []
    best_name, best_score = None, -float("inf")
    for name, model in MODELS.items():
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)
        if TASK_TYPE == "classification":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            auc = None
            if hasattr(model, "predict_proba") and len(set(y_test)) == 2:
                auc = roc_auc_score(y_test, model.predict_proba(X_test_t)[:, 1])
            metrics = {{"accuracy": round(acc, 4), "precision": round(
                precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "f1": round(f1, 4), "auc": round(auc, 4) if auc else None}}
            score = f1
        else:
            metrics = {{"r2": round(r2_score(y_test, y_pred), 4),
                        "mae": round(mean_absolute_error(y_test, y_pred), 4),
                        "mse": round(mean_squared_error(y_test, y_pred), 4),
                        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)}}
            score = metrics["r2"]
        results.append((name, metrics))
        print(f"    {{name:8s}} -> {{metrics}}")
        if score > best_score:
            best_name, best_score = name, score

    best = next(r for r in results if r[0] == best_name)
    print(f"[4/7] 最优模型：{{best_name}}")

    # ---- 可视化 ----
    best_model = MODELS[best_name]
    y_pred_best = best_model.predict(X_test_t)
    if TASK_TYPE == "classification":
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred_best, display_labels=class_names, cmap="Blues")
        disp.figure_.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
        print("[5/7] 已保存 confusion_matrix.png")
    else:
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, y_pred_best, alpha=0.6, edgecolors="k")
        lo = min(y_test.min(), y_pred_best.min()); hi = max(y_test.max(), y_pred_best.max())
        plt.plot([lo, hi], [lo, hi], "r--", label="y=x")
        plt.xlabel("真实值"); plt.ylabel("预测值")
        plt.title("真实值 vs 预测值（最优模型）")
        plt.legend(); plt.tight_layout()
        plt.savefig("scatter.png", dpi=120)
        print("[5/7] 已保存 scatter.png")

    # 特征重要性（随机森林）
    rf = next((m for n, m in MODELS.items() if "随机森林" in n), None)
    if rf is not None and hasattr(rf, "feature_importances_"):
        names = pre.get_feature_names_out()
        imp = rf.feature_importances_
        order = np.argsort(imp)[::-1][:15]
        plt.figure(figsize=(7, max(4, 0.4 * len(order))))
        plt.barh(range(len(order))[::-1], imp[order], color="#4C72B0")
        plt.yticks(range(len(order))[::-1], [str(names[i]) for i in order])
        plt.xlabel("重要性"); plt.title("特征重要性（随机森林）")
        plt.tight_layout(); plt.savefig("feature_importance.png", dpi=120)
        print("[6/7] 已保存 feature_importance.png")

    # ---- 指标汇总打印 ----
    print(f"[7/7] 最优模型 {{best_name}} 测试集指标：{{best[1]}}")
    print("全部结果见上方模型对比。")


if __name__ == "__main__":
    main()
'''
