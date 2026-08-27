# -*- coding: utf-8 -*-
"""数据探查接口：根据 file_id 读取表格，返回数据集概览 JSON。"""
from fastapi import APIRouter, HTTPException, Query

from app.config import UPLOAD_DIR
from app.ml.explore import explore_dataset, read_table

router = APIRouter(prefix="/api", tags=["数据探查"])


@router.get("/explore")
def explore(file_id: str = Query(..., description="上传接口返回的 file_id")):
    """读取上传的表格并返回：样本数、特征数、缺失统计、列类型、预览、候选标签列。"""
    # 防路径穿越：file_id 只允许字母数字，且只在 uploads 目录内查找
    if not file_id or not file_id.isalnum():
        raise HTTPException(status_code=400, detail="非法的 file_id")
    matched = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matched:
        raise HTTPException(status_code=404, detail="文件不存在，请先上传")

    try:
        df = read_table(matched[0])
    except Exception as e:  # 文件损坏/编码异常等统一转 422
        raise HTTPException(status_code=422, detail=f"表格读取失败：{e}")

    return explore_dataset(df)
