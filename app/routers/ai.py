# -*- coding: utf-8 -*-
"""AI 解读接口：调用 QClaw Agent 或本地规则，返回 Markdown 解读文本。"""
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import OUTPUT_DIR
from app.ml.ai_advisor import ai_interpret

router = APIRouter(prefix="/api", tags=["AI增强"])


@router.get("/ai/interpret/{task_id}")
def interpret(task_id: str):
    """对指定任务的结果做 AI 解读。"""
    if not task_id.isalnum():
        raise HTTPException(status_code=400, detail="非法的 task_id")
    path = OUTPUT_DIR / task_id / "result.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="结果不存在")
    result = __import__("json").loads(path.read_text(encoding="utf-8"))
    return {"task_id": task_id, "interpretation": ai_interpret(result)}
