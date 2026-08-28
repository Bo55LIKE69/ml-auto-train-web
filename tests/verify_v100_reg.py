# -*- coding: utf-8 -*-
"""回归任务（成交价）+ SHAP + PDF + 前端页面可用性验证"""
import io
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"
DATA = r"D:\ML_help\sample_data\房价预测.csv"


def main():
    out = io.StringIO()
    log = lambda *a: out.write(" ".join(str(x) for x in a) + "\n")
    c = httpx.Client(timeout=60)
    ok = total = 0

    def check(name, cond, extra=""):
        nonlocal ok, total
        total += 1
        if cond:
            ok += 1
        log(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

    # 回归训练（成交价）
    r = c.post(f"{BASE}/api/upload", files={"file": ("house.csv", open(DATA, "rb"), "text/csv")})
    fid = r.json().get("file_id")
    r = c.get(f"{BASE}/api/explore", params={"file_id": fid})
    suggested = r.json().get("suggested_target")
    log(f"    suggested_target = {suggested}")
    r = c.post(f"{BASE}/api/train", json={
        "file_id": fid, "target_col": "成交价", "task_type": "regression",
        "model_set": ["lr", "rf", "lightgbm"],
    })
    task_id = r.json().get("task_id")
    deadline = time.time() + 120
    while time.time() < deadline:
        s = c.get(f"{BASE}/api/tasks/{task_id}").json().get("status")
        if s in ("completed", "failed"):
            break
        time.sleep(2)
    check("regression train completed", s == "completed", f"status={s}")

    r = c.get(f"{BASE}/api/result/{task_id}")
    result = r.json()
    plots = result.get("plots", {})
    check("regression scatter present", "scatter" in plots)
    check("regression correlation present", "correlation" in plots)
    check("regression shap present", "shap_summary" in plots)
    if "shap_summary" in plots:
        r = c.get(BASE + plots["shap_summary"])
        check("regression shap download", r.status_code == 200 and r.content[:4] == b"\x89PNG",
              f"-> {r.status_code}")
    r = c.get(f"{BASE}/api/download/{task_id}/report.pdf")
    check("regression pdf", r.status_code == 200 and r.content[:4] == b"%PDF",
          f"-> {r.status_code} len={len(r.content)}")

    # 前端页面
    for page in ["/", "/select.html", "/result.html", "/dashboard.html",
                 "/css/style.css", "/js/dashboard.js", "/js/result.js"]:
        r = c.get(BASE + page)
        check(f"page {page}", r.status_code == 200 and len(r.content) > 500,
              f"-> {r.status_code} len={len(r.content)}")

    # dashboard 静态检查
    r = c.get(BASE + "/dashboard.html")
    html = r.text
    check("dashboard has table", "taskTable" in html and "新建训练任务" in html)
    r = c.get(BASE + "/js/dashboard.js")
    check("dashboard.js references api", "/api/tasks" in r.text)

    log(f"\n==== {ok}/{total} checks passed ====")
    print(out.getvalue())
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
