# -*- coding: utf-8 -*-
"""
全局配置模块：统一管理文件存储路径、允许的上传类型、任务目录等。
所有路径默认位于 D:/ML_help 下，可通过环境变量 ML_HELP_HOME 覆盖（便于换机器）。
"""
import os
from pathlib import Path

# 项目根目录：默认 D:/ML_help（可通过环境变量 ML_HELP_HOME 修改）
BASE_DIR = Path(os.environ.get("ML_HELP_HOME", r"D:/ML_help")).resolve()

# 上传文件存储目录
UPLOAD_DIR = BASE_DIR / "uploads"
# 训练产物目录（图表/报告/复现脚本，按 task_id 子目录隔离）
OUTPUT_DIR = BASE_DIR / "outputs"
# 前端静态文件目录
STATIC_DIR = BASE_DIR / "app" / "static"

# 允许上传的表格扩展名
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# 读取 CSV 时尝试的编码（中文 Windows 导出的 CSV 常为 GBK/GB18030）
CSV_ENCODINGS = ["utf-8", "gbk", "gb18030", "utf-8-sig"]

# 上传大小上限：50 MB
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

# 训练集划分比例与随机种子（保证结果可复现）
TEST_SIZE = 0.3
RANDOM_STATE = 42


def ensure_dirs():
    """确保运行时目录存在（幂等，启动时调用）。"""
    for d in (UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR):
        d.mkdir(parents=True, exist_ok=True)
