# -*- coding: utf-8 -*-
"""
运行环境自检接口：GET /api/deps

解决的问题：xgboost / lightgbm / catboost / shap / python-docx / joblib 等
可选依赖若未安装，程序会静默降级（少 3 个模型、没有 SHAP 图、无法导出 Word），
而界面上没有任何提示，学生根本不知道自己"少了功能"。

本接口把这些状态显式暴露出来，前端据此显示"运行环境"面板。
"""
from fastapi import APIRouter

from app.config import DEFAULT_MODEL_SET, check_optional_deps
from app.ml.train import model_status

router = APIRouter(prefix="/api", tags=["运行环境"])


@router.get("/deps")
def get_deps():
    """返回可选依赖与模型的可用性明细。"""
    deps = check_optional_deps()
    clf = model_status("classification", DEFAULT_MODEL_SET)
    reg = model_status("regression", DEFAULT_MODEL_SET)

    missing = [k for k, v in deps.items() if not v["available"]]
    return {
        "dependencies": deps,
        "models": {"classification": clf, "regression": reg},
        "missing_dependencies": missing,
        "all_ready": not missing,
        # 直接给出可复制的修复命令，学生不用自己查
        "install_command": (
            "pip install " + " ".join(deps[k]["pip_name"] for k in missing)
            if missing else None
        ),
    }
