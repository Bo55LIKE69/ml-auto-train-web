# -*- coding: utf-8 -*-
"""训练接口：接收任务参数，触发完整 ML 流水线，返回模型对比结果。"""
import json
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import OUTPUT_DIR, UPLOAD_DIR
from app.ml.explore import read_table
from app.ml.train import run_pipeline

router = APIRouter(prefix="/api", tags=["训练"])


class TrainRequest(BaseModel):
    """训练任务请求体。"""
    file_id: str = Field(..., description="上传接口返回的 file_id")
    target_col: str = Field(..., description="目标标签列名")
    task_type: str = Field("auto", description="auto / classification / regression")
    id_cols: list[str] = Field(default_factory=list, description="显式剔除的 ID 列")


@router.post("/train")
def train(req: TrainRequest):
    """触发训练：读取数据 → 预处理 → 多模型训练 → 评估 → 绘图 → 生成报告。"""
    if not req.file_id.isalnum():
        raise HTTPException(status_code=400, detail="非法的 file_id")
    matched = list(UPLOAD_DIR.glob(f"{req.file_id}.*"))
    if not matched:
        raise HTTPException(status_code=404, detail="文件不存在，请先上传")
    src = matched[0]

    try:
        df = read_table(src)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"表格读取失败：{e}")

    if req.target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"目标列 {req.target_col} 不存在，可选列：{list(df.columns)}",
        )

    # 每个训练任务一个独立输出目录 outputs/<task_id>/
    task_id = uuid.uuid4().hex[:12]
    out_dir = OUTPUT_DIR / task_id
    try:
        result = run_pipeline(
            df=df,
            target_col=req.target_col,
            task_type=req.task_type,
            id_cols=req.id_cols,
            out_dir=out_dir,
            source_file=str(src),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练失败：{e}")

    return result
