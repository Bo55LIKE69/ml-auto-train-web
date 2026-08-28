# -*- coding: utf-8 -*-
"""HTTP E2E 最终验证：12 全模型 + 回归 GIS 数据走 HTTP 链路。
手动运行：python tests/test_http_e2e_final.py
"""
import io
import json
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"
fails = []


def check(name, cond, extra=""):
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name} {extra}")
    if not cond:
        fails.append(name)


def wait_task(task_id, timeout=300):
    """轮询任务直到完成/失败，返回最终响应。"""
    lines, t0 = [], time.time()
    while time.time() - t0 < timeout:
        time.sleep(2)
        r = httpx.get(f"{BASE}/api/tasks/{task_id}", params={"log_cursor": len(lines)})
        d = r.json()
        lines.extend(d.get("log_lines", []))
        if d.get("status") in ("completed", "failed"):
            return d, lines
    return {"status": "timeout"}, lines


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("=== A. 分类任务：12 全模型 ===")
    with open(r"D:\ML_help\sample_data\学生成绩示例.csv", "rb") as f:
        r = httpx.post(f"{BASE}/api/upload", files={"file": ("学生成绩示例.csv", f, "text/csv")})
    file_id = r.json()["file_id"]
    check("A upload", r.status_code == 200, f"file_id={file_id}")

    r = httpx.post(f"{BASE}/api/train", json={
        "file_id": file_id, "target_col": "是否通过", "task_type": "classification",
        "id_cols": ["学号"],
    })
    check("A train submit", r.status_code == 200, f"({r.status_code})")
    task_id = r.json()["task_id"]
    d, lines = wait_task(task_id)
    check("A completed", d["status"] == "completed", f"status={d['status']}")
    check("A 12 models", d.get("metrics_summary", {}).get("total_models") == 12,
          f"models={d.get('metrics_summary', {}).get('total_models')}")
    if d["status"] == "completed":
        rj = httpx.get(f"{BASE}/api/result/{task_id}").json()
        check("A best", rj["best_model"]["name"] in [m["name"] for m in rj["models"]],
              f"best={rj['best_model']['name']}（12 模型中的最优）")
        # 验证 Word 报告
        r = httpx.get(f"{BASE}/api/download/{task_id}/report.docx")
        check("A docx", r.status_code == 200 and r.content[:2] == b"PK", f"({len(r.content)} bytes)")
        # 模型列表包含 boosting 家族
        names = [m["name"] for m in rj["models"]]
        check("A has XGB/LGBM/CatBoost", all(x in names for x in ("XGBoost", "LightGBM", "CatBoost")),
              f"names={names}")

    # ===== 测试 B：GIS 回归（12 模型自动裁剪为回归集）=====
    print("\n=== B. 回归任务：GIS 彩蛋数据集 ===")
    with open(r"D:\ML_help\demo-data\soil_gis.csv", "rb") as f:
        r = httpx.post(f"{BASE}/api/upload", files={"file": ("soil_gis.csv", f, "text/csv")})
    file_id = r.json()["file_id"]
    check("B upload", r.status_code == 200, f"file_id={file_id}")

    r = httpx.post(f"{BASE}/api/train", json={
        "file_id": file_id, "target_col": "yield", "task_type": "regression",
        "id_cols": ["site_id"],
    })
    check("B train submit", r.status_code == 200, f"({r.status_code})")
    task_id = r.json()["task_id"]
    d, lines = wait_task(task_id)
    check("B completed", d["status"] == "completed", f"status={d['status']}")
    if d["status"] == "completed":
        rj = httpx.get(f"{BASE}/api/result/{task_id}").json()
        check("B regression best", rj["best_model"]["name"] == "线性回归",
              f"best={rj['best_model']['name']}, R2={rj['best_model']['metrics'].get('r2')}")
        check("B has scatter", "scatter" in rj["plots"], f"plots={list(rj['plots'].keys())}")
        # 回归模型数：12 模型集裁剪后应 >= 9
        check("B reg models>=9", len(rj["models"]) >= 9, f"({len(rj['models'])} regressors)")
        # 回归任务下载散点图
        r = httpx.get(f"{BASE}/api/download/{task_id}/scatter.png")
        check("B scatter png", r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n",
              f"({r.status_code})")
        # IR 中 sort_metric 应为 R2
        ir = httpx.get(f"{BASE}/api/download/{task_id}/pipeline_ir.json").json()
        check("B ir sort R2", ir["setup_config"]["sort_metric"] == "R2",
              f"sort={ir['setup_config']['sort_metric']}")
        check("B ir fold_strategy", ir["setup_config"]["fold_strategy"] == "kfold",
              f"strategy={ir['setup_config']['fold_strategy']}")

    print(f"\n=== 结果: {len(fails)} 失败 ===")
    if fails:
        print("失败项:", fails)
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
