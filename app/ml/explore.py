# -*- coding: utf-8 -*-
"""
数据探查业务模块（纯 pandas 实现，不依赖 FastAPI）：
- read_table：读取 CSV/Excel 为 DataFrame（自动处理编码/中文列名）
- explore_dataset：返回数据集概览 dict（样本数、特征数、缺失统计、列类型、
  前几行预览、候选标签列建议）
"""
import math

import pandas as pd

from app.config import CSV_ENCODINGS


def read_table(file_path, sheet_name=0, **kwargs) -> pd.DataFrame:
    """
    读取表格文件为 DataFrame。
    - .csv：自动尝试 utf-8 / gbk / gb18030 / utf-8-sig 编码
    - .xlsx / .xls：使用 openpyxl / xlrd 读取，默认第一个工作表
    """
    path = str(file_path).lower()
    if path.endswith(".csv"):
        return _read_csv(file_path, **kwargs)
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
    raise ValueError(f"不支持的文件类型：{file_path}")


def _read_csv(file_path, **kwargs):
    """逐个编码尝试，直到能完整解码。"""
    last_err = None
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(file_path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def _safe_float(v):
    """把 NaN/Inf 转成 JSON 可序列化的 None。"""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def explore_dataset(df: pd.DataFrame) -> dict:
    """
    数据探查主函数：输入 DataFrame，输出概览字典（供 /api/explore 直接返回）。
    包含：样本数、特征数、各列类型、缺失统计、唯一值数、前 5 行预览、
    数值列清单、类别列清单、疑似 ID 列、候选标签列建议。
    """
    n_samples, n_features = df.shape
    col_info = []
    numeric_cols, categorical_cols = [], []
    id_like_cols = []

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()  # 按非空值推断，避免全缺失列误判为 float

        if pd.api.types.is_numeric_dtype(non_null):
            dtype = "数值型"
            numeric_cols.append(str(col))
        elif pd.api.types.is_datetime64_any_dtype(non_null):
            dtype = "日期型"
        else:
            dtype = "类别型"
            categorical_cols.append(str(col))

        missing = int(series.isna().sum())
        missing_rate = round(missing / n_samples, 4) if n_samples else 0.0
        is_id = _is_id_like(col, non_null, n_samples)

        if is_id:
            id_like_cols.append(str(col))

        col_info.append({
            "name": str(col),
            "dtype": dtype,
            "dtype_raw": str(series.dtype),
            "missing": missing,
            "missing_rate": missing_rate,
            "unique": int(non_null.nunique()),
            "is_id_like": is_id,
        })

    # 前 5 行预览（NaN -> None，保证 JSON 序列化成功）
    head = df.head(5).astype(object).where(pd.notnull(df.head(5)), None)
    preview = head.to_dict(orient="records")

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "columns": col_info,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "id_like_cols": id_like_cols,
        "missing_total": int(df.isna().sum().sum()),
        "preview": preview,
        "suggested_target": _suggest_target(df, categorical_cols, id_like_cols),
    }


def _is_id_like(col, non_null, n_samples) -> bool:
    """
    疑似 ID/编号列：仅用列名关键词匹配（id/编号/序号/学号/工号/卡号）。
    注意：不使用"唯一率>95%"启发式，避免把成绩等真实特征误判为 ID。
    """
    name = str(col).lower()
    keywords = ("id", "编号", "序号", "index", "学号", "工号", "卡号")
    return any(k in name for k in keywords)


def _suggest_target(df: pd.DataFrame, categorical_cols, id_like_cols=None) -> str | None:
    """
    候选标签列建议（供前端下拉框默认选中，用户仍可手动改选任意列）：
    1. 先排除 ID/编号类无意义列；
    2. 列名含 是否/结果/标签/目标/类别/class/label/target 等关键词的优先（多为标签）；
    3. 否则在取值数 2~20 的类别列中选取值数最少者，并列时取位置靠后的列
       （多数数据集标签位于最后一列）。
    """
    cand = [c for c in categorical_cols if not (id_like_cols and c in id_like_cols)]
    if not cand:
        return None

    keywords = ("是否", "结果", "标签", "目标", "类别", "class", "label", "target")
    for c in cand:
        if any(k in str(c).lower() for k in keywords):
            return c

    scored = [(int(df[c].nunique()), i, c) for i, c in enumerate(cand)]
    mid = [s for s in scored if 2 <= s[0] <= 20]  # 取值数适中者优先
    pool = mid if mid else scored
    # nunique 最小优先；并列时 -i 最小 => 位置靠后的列胜出
    return min(pool, key=lambda s: (s[0], -s[1]))[2]
