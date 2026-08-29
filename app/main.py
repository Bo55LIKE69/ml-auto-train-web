# -*- coding: utf-8 -*-
"""
FastAPI 应用入口。
启动：uvicorn app.main:app --reload --port 8000
接口文档：http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR, ensure_dirs
from app.routers import ai, download, explore, predict, result, tasks, train, upload

# 创建运行时目录（uploads / outputs / static）
ensure_dirs()

app = FastAPI(
    title="表格数据集机器学习自动训练工具",
    description="上传 CSV/Excel → 自动探查、预处理、多模型训练对比、可视化与报告生成（含 SHAP 可解释性 / PDF 报告）",
    version="1.0.0",
)

# 开发期放开跨域（前端若用 file:// 直接打开或换端口访问时需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册接口路由（必须在静态挂载之前注册，否则会被 mount("/") 吞掉）
app.include_router(upload.router)
app.include_router(explore.router)
app.include_router(train.router)
app.include_router(tasks.router)
app.include_router(predict.router)
app.include_router(download.router)
app.include_router(ai.router)
app.include_router(result.router)


@app.get("/api/health")
def health():
    """健康检查。"""
    return {"status": "ok", "service": "表格ML自动训练工具"}

# 静态页面（模块3 提供 index.html 后，访问 / 即打开上传页）
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
