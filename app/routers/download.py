# -*- coding: utf-8 -*-
"""下载接口：提供训练产物下载（图表 / 报告 / 复现脚本 / 结果JSON）。"""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import OUTPUT_DIR

router = APIRouter(prefix="/api/download", tags=["下载"])

# 允许下载的产物类型与文件名映射
ALLOWED_ARTIFACTS = {
    "report.md": "text/markdown; charset=utf-8",
    "script.py": "text/x-python; charset=utf-8",
    "result.json": "application/json",
    "confusion_matrix.png": "image/png",
    "scatter.png": "image/png",
    "feature_importance.png": "image/png",
}


@router.get("/{task_id}/{artifact}")
def download(task_id: str, artifact: str):
    """下载指定任务的产物文件。task_id 与 artifact 均做了白名单校验，防路径穿越。"""
    if not (task_id.isalnum() and artifact in ALLOWED_ARTIFACTS):
        raise HTTPException(status_code=400, detail="非法请求参数")
    path = OUTPUT_DIR / task_id / artifact
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path,
        media_type=ALLOWED_ARTIFACTS[artifact],
        filename=f"{task_id}_{artifact}",
    )
