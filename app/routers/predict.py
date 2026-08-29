# -*- coding: utf-8 -*-
"""
在线预测接口：加载已训练任务持久化的最优模型，对新数据做推理。
规格书补充：POST /api/predict/{task_id}
学生训完模型后，可上传同结构新数据，复用已 fit 的 preprocessor + 最优模型做预测。
"""
import io
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import OUTPUT_DIR
from app.ml.explore import read_table

router = APIRouter(prefix="/api", tags=["预测"])

# 允许下载的预测产物类型
_PREDICT_MEDIA = "text/csv; charset=utf-8"


@router.post("/predict/{task_id}")
async def predict(task_id: str, file: UploadFile = File(...)):
    """对 task_id 对应的已训模型，预测上传的新数据。"""
    if not task_id.isalnum():
        raise HTTPException(status_code=400, detail="非法 task_id")
    art_path = OUTPUT_DIR / task_id / "model_artifacts.joblib"
    if not art_path.is_file():
        raise HTTPException(status_code=404,
                            detail="该任务未持久化模型文件，请重新训练（需保持默认设置）")

    # 读取新数据
    try:
        raw = await file.read()
        suffix = Path(file.filename).suffix.lower() if file.filename else ".csv"
        if suffix in (".xlsx", ".xls"):
            new_df = pd.read_excel(io.BytesIO(raw))
        else:
            new_df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"新数据读取失败：{e}")

    try:
        art = joblib.load(art_path)
        preprocessor = art["preprocessor"]
        model = art["model"]
        target_col = art.get("target_col")
        class_names = art.get("class_names")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型加载失败：{e}")

    # 列对齐：新数据应包含训练时所用特征（允许多余列，剔除目标列与未用列）
    try:
        X_new = new_df.drop(columns=[target_col], errors="ignore")
        X_new_t = preprocessor.transform(X_new)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"新数据与训练特征不匹配（需包含训练时的特征列）：{e}",
        )

    try:
        preds = model.predict(X_new_t)
        if class_names is not None and len(class_names):
            preds = [class_names[int(p)] if 0 <= int(p) < len(class_names) else p
                     for p in preds]
        out = new_df.copy()
        out["预测结果"] = preds
        # 若模型支持概率/置信度，附第一列概率（二分类/多分类）
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_new_t)
                out["预测置信度"] = [round(float(max(row)), 4) for row in proba]
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败：{e}")

    # 落盘预测结果 CSV，供下载（固定文件名，多次预测覆盖，保持 download 白名单稳定）
    pred_path = OUTPUT_DIR / task_id / "predictions.csv"
    out.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 预览：把 NaN/inf 清洗为 None，确保 JSON 可序列化
    import math as _math
    import numpy as _np

    def _clean(v):
        if v is None:
            return None
        if isinstance(v, (_np.integer,)):
            return int(v)
        if isinstance(v, (_np.floating,)):
            f = float(v)
            return None if (_math.isnan(f) or _math.isinf(f)) else f
        if isinstance(v, float):
            return None if (_math.isnan(v) or _math.isinf(v)) else v
        if isinstance(v, (_np.bool_,)):
            return bool(v)
        return v

    preview_raw = out.head(200).to_dict(orient="records")
    preview = [{k: _clean(v) for k, v in row.items()} for row in preview_raw]
    return {
        "task_id": task_id,
        "model_name": art.get("model_name"),
        "n_predicted": int(len(out)),
        "download_url": f"/api/download/{task_id}/predictions.csv",
        "preview": preview,
        "columns": list(out.columns),
    }
