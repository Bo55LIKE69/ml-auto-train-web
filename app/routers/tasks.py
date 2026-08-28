# -*- coding: utf-8 -*-
"""
历史任务列表接口：GET /api/tasks（支持 ?limit= / ?status=）
返回所有已完成/失败任务的精简摘要，供前端 Dashboard 展示。
"""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.config import OUTPUT_DIR

router = APIRouter(prefix="/api", tags=["任务列表"])


def _task_summary(task_dir: Path) -> dict:
    """从 task_dir 读取 result.json，生成精简摘要。"""
    rj = (task_dir / "result.json").read_text(encoding="utf-8")
    result = json.loads(rj)
    bm = result.get("best_model", {})
    metrics = bm.get("metrics", {})
    score_key = "f1" if result.get("task_type") == "classification" else "r2"
    # 时间取产物文件 mtime
    mtime = (task_dir / "result.json").stat().st_mtime
    return {
        "task_id": result.get("task_id", task_dir.name),
        "target_col": result.get("target_col"),
        "task_type": result.get("task_type"),
        "n_samples": result.get("n_samples"),
        "n_models": len(result.get("models", [])),
        "best_model": bm.get("name"),
        "best_score": metrics.get(score_key),
        "score_key": score_key,
        "total_time_sec": result.get("training_time_total_seconds"),
        "created_at": int(mtime),
        "has_pdf": (task_dir / "report.pdf").is_file(),
        "has_docx": (task_dir / "report.docx").is_file(),
        "status": "completed",
    }


@router.get("/tasks")
def list_tasks(limit: int = Query(50, ge=1, le=200),
               status: str = Query(None, description="completed/failed")):
    """列出历史任务（按创建时间倒序）。"""
    if not OUTPUT_DIR.exists():
        return []
    summaries = []
    for d in OUTPUT_DIR.iterdir():
        if not d.is_dir():
            continue
        rj = d / "result.json"
        if rj.is_file():
            try:
                summaries.append(_task_summary(d))
            except Exception:
                continue
        # 失败的任务（无 result.json，但有 training.log 且非训练状态）
        elif (d / "training.log").is_file():
            failed = _failed_summary(d)
            if failed:
                summaries.append(failed)
    # 按时间倒序
    summaries.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    if status:
        summaries = [s for s in summaries if s.get("status") == status]
    return summaries[:limit]


def _failed_summary(task_dir: Path) -> dict:
    """读取失败任务的 training.log 末行作为错误信息。"""
    try:
        log_text = (task_dir / "training.log").read_text(encoding="utf-8", errors="replace")
        last_err = ""
        for line in log_text.splitlines():
            if "失败" in line or "Error" in line or "error" in line:
                last_err = line.strip()
        mtime = (task_dir / "training.log").stat().st_mtime
        return {
            "task_id": task_dir.name,
            "status": "failed",
            "error_message": last_err or "训练失败（详情见日志）",
            "created_at": int(mtime),
        }
    except Exception:
        return None
