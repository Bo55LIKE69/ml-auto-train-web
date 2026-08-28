# -*- coding: utf-8 -*-
"""
下载接口：提供训练产物下载（图表 / 报告 / 复现脚本 / 结果JSON / Word报告）。
规格书 §6.5 GET /api/download/{task_id}/{file_type}
"""
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import OUTPUT_DIR

router = APIRouter(prefix="/api/download", tags=["下载"])

# 允许下载的产物类型与文件名映射
ALLOWED_ARTIFACTS = {
    "report.md": "text/markdown; charset=utf-8",
    "report.docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "report.pdf": "application/pdf",
    "script.py": "text/x-python; charset=utf-8",
    "result.json": "application/json",
    "confusion_matrix.png": "image/png",
    "scatter.png": "image/png",
    "feature_importance.png": "image/png",
    "metrics_comparison.png": "image/png",
    "correlation.png": "image/png",
    "shap_summary.png": "image/png",
    "pipeline_ir.json": "application/json",
    "training.log": "text/plain; charset=utf-8",
}


def _make_zip(task_dir: Path, files: list, zip_name: str) -> Path:
    """将指定文件打包为 zip（规格书 §6.5 all/charts 下载）。"""
    zip_path = task_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.is_file():
                zf.write(f, arcname=f.name)
    return zip_path


@router.get("/{task_id}/{artifact}")
def download(task_id: str, artifact: str):
    """下载指定任务的产物文件。task_id 与 artifact 均做了白名单校验，防路径穿越。"""
    if not task_id.isalnum():
        raise HTTPException(status_code=400, detail="非法 task_id")
    task_dir = OUTPUT_DIR / task_id

    # 打包下载（规格书 §6.5）：charts / all
    if artifact == "charts":
        chart_files = [p for p in task_dir.glob("*.png")]
        if not chart_files:
            raise HTTPException(status_code=404, detail="没有图表产物")
        zip_path = _make_zip(task_dir, chart_files, "charts.zip")
        return FileResponse(zip_path, media_type="application/zip",
                            filename=f"{task_id}_charts.zip")
    if artifact == "all":
        files = [p for p in task_dir.iterdir() if p.is_file() and p.suffix != ".zip"]
        zip_path = _make_zip(task_dir, files, "all_artifacts.zip")
        return FileResponse(zip_path, media_type="application/zip",
                            filename=f"{task_id}_all.zip")

    if artifact not in ALLOWED_ARTIFACTS:
        raise HTTPException(status_code=400, detail="非法产物类型")
    # PDF 报告：若不存在则尝试用 LibreOffice 现转（docx -> pdf）
    if artifact == "report.pdf":
        pdf_path = task_dir / "report.pdf"
        if not pdf_path.is_file():
            docx_path = task_dir / "report.docx"
            if not docx_path.is_file():
                raise HTTPException(status_code=404, detail="Word 报告不存在，无法转换 PDF")
            from app.ml.pdf_export import convert_docx_to_pdf
            ok = convert_docx_to_pdf(docx_path, pdf_path)
            if not ok:
                raise HTTPException(status_code=500, detail="PDF 转换失败（需要本机安装 LibreOffice）")
    path = task_dir / artifact
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path,
        media_type=ALLOWED_ARTIFACTS[artifact],
        filename=f"{task_id}_{artifact}",
    )
