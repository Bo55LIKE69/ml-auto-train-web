# -*- coding: utf-8 -*-
"""
训练接口：接收任务参数，触发完整 ML 流水线，返回模型对比结果。
规格书 §6.3 POST /api/train + §6.4 GET /api/tasks/{task_id}
"""
import json
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.config import (DEFAULT_FE_OPTS, DEFAULT_FOLD, DEFAULT_MODEL_SET,
                        OUTPUT_DIR, UPLOAD_DIR)
from app.ml.explore import read_table
from app.ml.train import (LogCapture, MODEL_CATALOG, run_pipeline)

router = APIRouter(prefix="/api", tags=["训练"])

# 后台任务注册表：{task_id: {status, result, log: LogCapture, error}}
# 注意：仅存内存，服务重启即清空 —— 因此所有状态变更都会同步落盘到
# outputs/<task_id>/status.json，重启后仍可查询（见 recover_interrupted_tasks）。
_TASKS = {}
_LOCK = threading.Lock()


def _write_status(task_id: str, status: str, **extra):
    """将任务状态落盘，保证服务重启后仍可查询。失败不影响主流程。"""
    try:
        out_dir = OUTPUT_DIR / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        data = {"task_id": task_id, "status": status,
                "updated_at": time.time(), **extra}
        (out_dir / "status.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _read_status(task_id: str):
    """读取落盘的任务状态，不存在返回 None。"""
    sf = OUTPUT_DIR / task_id / "status.json"
    if not sf.is_file():
        return None
    try:
        return json.loads(sf.read_text(encoding="utf-8"))
    except Exception:
        return None


def recover_interrupted_tasks() -> int:
    """
    服务启动时调用：把上次遗留的 training 状态标记为 interrupted。
    否则这些任务会永远显示"训练中"，前端一直转圈，学生以为程序卡死。
    返回被标记的任务数。
    """
    if not OUTPUT_DIR.exists():
        return 0
    n = 0
    for d in OUTPUT_DIR.iterdir():
        if not d.is_dir():
            continue
        sf = d / "status.json"
        if not sf.is_file():
            continue
        try:
            st = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if st.get("status") == "training":
            st["status"] = "interrupted"
            st["error"] = "服务重启导致训练中断，请重新上传数据并提交训练"
            st["updated_at"] = time.time()
            try:
                sf.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
                n += 1
            except Exception:
                pass
    return n


class TrainRequest(BaseModel):
    """训练任务请求体。"""
    file_id: str = Field(..., description="上传接口返回的 file_id")
    target_col: str = Field(..., description="目标标签列名")
    task_type: str = Field("auto", description="auto / classification / regression")
    id_cols: list[str] = Field(default_factory=list, description="显式剔除的 ID 列")
    model_set: list[str] = Field(default_factory=list, description="模型缩写列表，空则用默认 12 模型")
    fold: int = Field(DEFAULT_FOLD, description="交叉验证折数")
    sort_metric: str = Field("F1", description="排序指标（F1 / R2）")
    fe_opts: dict = Field(default_factory=dict, description="特征工程选项（缺失值策略/缩放/类别编码）")
    tune: bool = Field(False, description="是否对最优模型做超参调优（RandomizedSearchCV）")
    tune_budget: int = Field(20, description="RandomizedSearchCV 迭代次数上限（仅 tune=true 时生效）")
    # None 表示跟随服务端默认（config.CV_ENABLED）；前端可显式关掉以缩短耗时
    use_cv: Optional[bool] = Field(
        None,
        description="是否启用 K 折交叉验证；不传则跟随服务端默认配置",
    )


@router.get("/models")
def list_models():
    """返回可选模型清单（分类/回归），供前端手动选模型。"""
    return {
        "classification": [
            {"key": k, "name": v["name"], "desc": v.get("desc", "")}
            for k, v in MODEL_CATALOG["classification"].items()
        ],
        "regression": [
            {"key": k, "name": v["name"], "desc": v.get("desc", "")}
            for k, v in MODEL_CATALOG["regression"].items()
        ],
    }


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

    model_set = list(req.model_set) if req.model_set else list(DEFAULT_MODEL_SET)
    task_id = uuid.uuid4().hex[:12]
    out_dir = OUTPUT_DIR / task_id
    log = LogCapture()

    def _run():
        try:
            result = run_pipeline(
                df=df,
                target_col=req.target_col,
                task_type=req.task_type,
                id_cols=req.id_cols,
                out_dir=out_dir,
                source_file=str(src),
                model_set=model_set,
                fold=req.fold,
                log=log,
                fe_opts=req.fe_opts or None,
                tune=req.tune,
                tune_budget=req.tune_budget,
                use_cv=req.use_cv,
            )
            _write_status(task_id, "completed",
                          best_model=result["best_model"]["name"])
            with _LOCK:
                _TASKS[task_id] = {"status": "completed", "result": result, "log": log}
        except Exception as e:
            _write_status(task_id, "failed", error=str(e))
            with _LOCK:
                _TASKS[task_id] = {"status": "failed", "error": str(e), "log": log}

    with _LOCK:
        _TASKS[task_id] = {"status": "training", "log": log}
    _write_status(task_id, "training", target_col=req.target_col)
    threading.Thread(target=_run, daemon=True).start()

    return {
        "task_id": task_id,
        "status": "training",
        "message": f"训练已启动，共 {len(model_set)} 个模型",
        "estimated_seconds": min(60, 5 + len(model_set) * 4),
    }


@router.get("/tasks/{task_id}")
def get_task(task_id: str, log_cursor: int = 0):
    """轮询训练状态（规格书 §6.4）：返回增量日志 + 进度。"""
    with _LOCK:
        task = _TASKS.get(task_id)
    if task is None:
        # 服务重启后内存状态已清空：先查落盘的 status.json，再查 result.json
        out_dir = OUTPUT_DIR / task_id
        if (out_dir / "result.json").exists():
            result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
            return {
                "task_id": task_id, "status": "completed",
                "result_available": True, "metrics_summary": {
                    "best_model": result["best_model"]["name"],
                    "best_score": result["best_model"]["metrics"].get("f1", result["best_model"]["metrics"].get("r2")),
                    "total_models": len(result["models"]),
                    "total_time_sec": result["training_time_total_seconds"],
                },
                "log_lines": [], "next_cursor": 0,
            }
        st = _read_status(task_id)
        if st is not None:
            # interrupted（重启遗留）/ failed：明确告知，避免前端无限转圈
            status = st.get("status", "unknown")
            return {
                "task_id": task_id,
                "status": status,
                "result_available": False,
                "log_lines": [],
                "next_cursor": 0,
                "error": {
                    "code": f"TASK_{status.upper()}",
                    "message": st.get("error") or (
                        "任务未产生结果，请重新提交训练" if status == "interrupted"
                        else "任务已结束"),
                },
            }
        raise HTTPException(status_code=404, detail="任务不存在")

    log = task["log"]
    lines = log.get_lines_since(log_cursor)
    status = task["status"]
    resp = {
        "task_id": task_id,
        "status": status,
        "log_lines": lines,
        "next_cursor": log_cursor + len(lines),
    }
    if status == "completed":
        result = task["result"]
        resp["result_available"] = True
        resp["metrics_summary"] = {
            "best_model": result["best_model"]["name"],
            "best_score": result["best_model"]["metrics"].get("f1", result["best_model"]["metrics"].get("r2")),
            "total_models": len(result["models"]),
            "total_time_sec": result["training_time_total_seconds"],
        }
        resp["result_url"] = f"/api/result/{task_id}"
    elif status == "failed":
        resp["error"] = {"code": "TRAIN_FAILED", "message": task["error"]}
    return resp
