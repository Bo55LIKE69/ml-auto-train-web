# -*- coding: utf-8 -*-
"""结果查询接口：GET /api/result/{task_id} 返回完整训练结果 JSON。"""
import json

from fastapi import APIRouter, HTTPException

from app.config import OUTPUT_DIR

router = APIRouter(prefix="/api", tags=["结果"])


@router.get("/result/{task_id}")
def get_result(task_id: str):
    """返回完整训练结果（result.json）。"""
    if not task_id.isalnum():
        raise HTTPException(status_code=400, detail="非法 task_id")
    path = OUTPUT_DIR / task_id / "result.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="结果不存在")
    return json.loads(path.read_text(encoding="utf-8"))
