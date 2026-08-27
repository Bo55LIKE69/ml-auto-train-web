# -*- coding: utf-8 -*-
"""文件上传接口：接收 CSV/Excel，校验后落盘 uploads/，返回 file_id。"""
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE, UPLOAD_DIR

router = APIRouter(prefix="/api", tags=["上传"])


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传表格文件（CSV / XLSX / XLS）。
    返回 file_id，后续探查/训练接口均使用该 ID 定位文件。
    """
    # 1. 校验扩展名
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持 {sorted(ALLOWED_EXTENSIONS)} 格式的文件",
        )

    # 2. 生成唯一 file_id 并分块落盘（保留原扩展名，防大文件占内存）
    file_id = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    size = 0
    try:
        with open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB 一块
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="文件超过 50MB 上限")
                f.write(chunk)
    except HTTPException:
        save_path.unlink(missing_ok=True)  # 超限时删除半成品文件
        raise

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": size,
        "saved_path": str(save_path),
        "next": "/api/explore",  # 下一步接口提示
    }
