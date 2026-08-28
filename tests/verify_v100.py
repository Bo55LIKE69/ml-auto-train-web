# -*- coding: utf-8 -*-
"""
v1.0.0 端到端验证脚本：
1. 上传 CSV
2. 探索
3. 训练（少量模型加速，但含树模型以触发 SHAP）
4. 验证产物：correlation.png / shap_summary.png / metrics_comparison.png
5. 验证 plot URL 修复（/api/download/{task_id}/correlation.png 返回 200）
6. 验证 PDF 转换（/api/download/{task_id}/report.pdf 返回 200）
7. 验证 /api/tasks 包含新任务
"""
import io
import json
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"
DATA = r"D:\ML_help\sample_data\客户流失预测.csv"


def main():
    out = io.StringIO()
    log = lambda *a: out.write(" ".join(str(x) for x in a) + "\n")

    c = httpx.Client(timeout=60)
    ok = 0
    total = 0

    def check(name, cond, extra=""):
        nonlocal ok, total
        total += 1
        mark = "PASS" if cond else "FAIL"
        if cond:
            ok += 1
        log(f"[{mark}] {name} {extra}")

    # 1. upload
    r = c.post(f"{BASE}/api/upload", files={"file": ("churn.csv", open(DATA, "rb"), "text/csv")})
    check("upload", r.status_code == 200, f"-> {r.status_code}")
    fid = r.json().get("file_id")
    check("upload file_id", bool(fid), fid or "")

    # 2. explore
    r = c.get(f"{BASE}/api/explore", params={"file_id": fid})
    check("explore", r.status_code == 200, f"-> {r.status_code}")
    suggested = r.json().get("suggested_target")

    # 3. train（lr + rf + lightgbm，含树模型触发 SHAP）
    r = c.post(f"{BASE}/api/train", json={
        "file_id": fid,
        "target_col": suggested,
        "task_type": "classification",
        "model_set": ["lr", "rf", "lightgbm"],
    })
    check("train submit", r.status_code == 200, f"-> {r.status_code}")
    task_id = r.json().get("task_id")
    check("train task_id", bool(task_id), task_id or "")

    # 4. poll
    deadline = time.time() + 120
    status = None
    while time.time() < deadline:
        r = c.get(f"{BASE}/api/tasks/{task_id}")
        status = r.json().get("status")
        if status in ("completed", "failed"):
            break
        time.sleep(2)
    check("train completed", status == "completed", f"status={status}")

    # 5. result.json
    r = c.get(f"{BASE}/api/result/{task_id}")
    check("result.json", r.status_code == 200, f"-> {r.status_code}")
    result = r.json()
    plots = result.get("plots", {})
    log(f"    plots keys: {list(plots.keys())}")
    check("correlation plot present", "correlation" in plots)
    check("shap plot present", "shap_summary" in plots)
    check("metrics_comparison present", "metrics_comparison" in plots)

    # 6. 单图 URL 可直接下载（原 bug：/plot/ 前缀导致 404）
    if "correlation" in plots:
        url = plots["correlation"]
        r = c.get(BASE + url)
        check("plot url direct download", r.status_code == 200 and r.content[:4] == b"\x89PNG",
              f"-> {r.status_code} len={len(r.content)} url={url}")
    if "shap_summary" in plots:
        url = plots["shap_summary"]
        r = c.get(BASE + url)
        check("shap url direct download", r.status_code == 200 and r.content[:4] == b"\x89PNG",
              f"-> {r.status_code} len={len(r.content)}")

    # 7. PDF 转换（LibreOffice）
    r = c.get(f"{BASE}/api/download/{task_id}/report.pdf")
    check("pdf report", r.status_code == 200 and r.content[:4] == b"%PDF",
          f"-> {r.status_code} len={len(r.content)}")

    # 8. /api/tasks 包含新任务
    r = c.get(f"{BASE}/api/tasks?limit=100")
    tasks = r.json()
    check("tasks list contains new", any(t["task_id"] == task_id for t in tasks),
          f"total tasks={len(tasks)}")

    # 9. charts zip 含新图
    r = c.get(f"{BASE}/api/download/{task_id}/charts")
    check("charts zip", r.status_code == 200 and r.content[:2] == b"PK",
          f"-> {r.status_code} len={len(r.content)}")

    log(f"\n==== {ok}/{total} checks passed ====")
    print(out.getvalue())
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
