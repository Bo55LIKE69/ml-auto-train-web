# -*- coding: utf-8 -*-
"""
数据预处理业务模块（纯 pandas/sklearn 实现，不依赖 FastAPI，可独立测试）。

核心函数 prepare_dataset：
  1. 校验目标列，剔除目标列缺失的样本行
  2. 剔除 ID/编号类无意义列（前端传入 + 自动检测：列名关键词 或 唯一率>95% 的整型列）
  3. 剔除日期列、高基数类别列（>50 个取值，多为自由文本，建模意义低）
  4. 自动判定任务类型（分类 / 回归）
  5. 构建预处理管道：数值列 → 中位数填充 + 标准化；
     类别列 → 众数填充 + 独热编码

重要：返回的管道【不在这里 fit】，由训练模块在训练集上 fit，
避免测试集信息泄漏到训练过程（论文答辩要点）。
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 类别列取值上限：超过则视为高基数文本列，直接剔除
HIGH_CARDINALITY_LIMIT = 50
# ID 列关键词（中文列名大小写不敏感）
ID_KEYWORDS = ("id", "编号", "序号", "index", "学号", "工号", "卡号", "账号")


def detect_task_type(y: pd.Series) -> str:
    """自动判定任务类型：数值型目标且取值数>20 视为回归，否则为分类。"""
    yy = y.dropna()
    if pd.api.types.is_numeric_dtype(yy) and yy.nunique() > 20:
        return "regression"
    return "classification"


def _looks_like_id(col: str, series: pd.Series) -> bool:
    """
    ID 列检测：仅用列名关键词匹配（学号/编号/id/序号等）。
    注意：不使用"唯一率>95%"启发式——小样本数据集中，成绩/分数等真实
    特征列也常 100% 唯一，自动剔除会误杀有效特征。
    用户仍可在前端手动勾选额外剔除列。
    """
    name = str(col).lower()
    return any(k in name for k in ID_KEYWORDS)


def build_preprocessor(df: pd.DataFrame):
    """
    按列类型构建 ColumnTransformer：
    - 数值列：SimpleImputer(中位数) → StandardScaler
    - 类别列：SimpleImputer(众数) → OneHotEncoder(handle_unknown='ignore')
    - remainder='drop'：其余列（理论上不存在）直接丢弃
    返回 (preprocessor, num_cols, cat_cols)
    """
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dropna())]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c].dropna())]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),  # 数值列缺失填中位数
                ("scaler", StandardScaler()),                    # 标准化（线性模型/SVM 需要）
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),  # 类别列缺失填众数
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))
    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def prepare_dataset(df: pd.DataFrame, target_col: str, id_cols=None):
    """
    数据预处理入口。

    参数：
        df        原始 DataFrame
        target_col 目标标签列名
        id_cols   用户显式指定要剔除的列（可选）

    返回：
        X           剔除无用列后的特征 DataFrame（未填充缺失）
        y           目标列原始 Series（已剔除目标缺失行）
        y_encoder   占位（None，由训练模块按任务类型决定是否编码）
        preprocessor 未 fit 的 ColumnTransformer
        meta        元信息 dict：task_type / warnings / drop_cols / 列清单等
    """
    if target_col not in df.columns:
        raise ValueError(f"目标列 {target_col} 不存在")

    df = df.copy()
    y = df[target_col]
    X = df.drop(columns=[target_col])
    warnings = []

    # 1) 剔除目标列缺失的行
    mask = y.notna()
    if not mask.all():
        warnings.append(f"目标列存在 {int((~mask).sum())} 行缺失，已剔除这些样本")
        X, y = X[mask], y[mask]
    if len(X) == 0:
        raise ValueError("目标列全部缺失，无法训练")

    # 2) 剔除 ID/编号类无意义列（显式传入 + 自动检测）
    drop_cols = []
    for c in (id_cols or []):
        if c in X.columns:
            drop_cols.append(c)
    for c in X.columns:
        if c not in drop_cols and _looks_like_id(c, X[c]):
            drop_cols.append(c)
    if drop_cols:
        warnings.append(f"已剔除无意义 ID/编号列：{drop_cols}")
        X = X.drop(columns=drop_cols)

    # 3) 剔除日期列（本系统暂不做时间特征工程）
    datetime_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
    if datetime_cols:
        warnings.append(f"已剔除日期列（暂不支持时间特征）：{datetime_cols}")
        X = X.drop(columns=datetime_cols)

    # 4) 剔除高基数类别列（>50 取值，多为自由文本）
    high_card_cols = []
    for c in X.columns:
        s = X[c].dropna()
        if len(s) and not pd.api.types.is_numeric_dtype(s) and s.nunique() > HIGH_CARDINALITY_LIMIT:
            high_card_cols.append(c)
    if high_card_cols:
        warnings.append(f"已剔除高基数文本列（取值>50，建模意义低）：{high_card_cols}")
        X = X.drop(columns=high_card_cols)

    # 5) 样本量检查（风险提示）
    n = len(X)
    if n < 30:
        warnings.append("! 样本数不足 30，模型结果可靠性很低，请补充数据")
    elif n < 100:
        warnings.append("! 样本数少于 100，结果存在一定偶然性，建议补充数据")

    # 6) 缺失率检查（>30% 提示）
    high_missing = X.isna().mean()
    high_missing = high_missing[high_missing > 0.3]
    for c, r in high_missing.items():
        warnings.append(f"! 特征 [{c}] 缺失率达 {r:.0%}，已自动填充，建议核实数据来源")

    # 7) 任务类型（用户可在前端强制覆盖为 classification/regression）
    task_type = detect_task_type(y)

    # 8) 构建预处理管道（未 fit）
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    meta = {
        "task_type": task_type,
        "warnings": warnings,
        "drop_cols": drop_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "n_samples": n,
        "n_features_raw": X.shape[1],
    }
    return X, y, None, preprocessor, meta
