# -*- coding: utf-8 -*-
"""v1.0.0 全模型 E2E v2：带诊断输出"""
import io
import sys
import time
import traceback

import httpx

BASE = "http://127.0.0.1:8000"


def run_one(c, name, data_path, target_col, task_type, model_set, id_cols=None):
    out = []
    try:
        r = c.post(f"{BASE}/api/upload", files={"file": (name, open(data_path, "rb"), "text/csv")})
        out.append(f"upload: {r.status_code}")
        if r.status_code != 200:
            out.append(f"  body: {r.text[:200]}")
            return "\n".join(out), False
        fid = r.json().get("file_id")
        body = {"file_id": fid, "target_col": target_col, "task_type": task_type,
                "model_set": model_set, "id_cols": id_cols or []}
        r = c.post(f"{BASE}/api/train", json=body)
        out.append(f"train: {r.status_code} {r.text[:150]}")
        if r.status_code != 200:
            return "\n".join(out), False
        task_id = r.json().get("task_id")
        if not task_id:
            out.append("NO TASK_ID!")
            return "\n".join(out), False
        deadline = time.time() + 300
        s = "training"
        while time.time() < deadline:
            r = c.get(f"{BASE}/api/tasks/{task_id}")
            s = r.json().get("status")
            if s in ("completed", "failed"):
                break
            time.sleep(3)
        out.append(f"{name}: status={s} task={task_id}")
        if s != "completed":
            return "\n".join(out), False
        r = c.get(f"{BASE}/api/result/{task_id}")
        res = r.json()
        plots = res.get("plots", {})
        out.append(f"  best={res['best_model']['name']}")
        out.append(f"  plots={list(plots.keys())}")
        out.append(f"  models={len(res['models'])}")
        if "shap_summary" in plots:
            r = c.get(BASE + plots["shap_summary"])
            ok_shap = r.status_code == 200 and r.content[:4] == b"\x89PNG"
            out.append(f"  shap download: {'OK' if ok_shap else 'FAIL'} ({len(r.content)}B)")
        else:
            ok_shap = False
            out.append("  shap MISSING!")
        return "\n".join(out), ok_shap
    except Exception as e:
        out.append(f"EXCEPTION: {type(e).__name__}: {e}")
        out.append(traceback.format_exc()[-800:])
        return "\n".join(out), False


def main():
    out = io.StringIO()
    log = lambda *a: out.write(" ".join(str(x) for x in a) + "\n")
    c = httpx.Client(timeout=120)
    ok = total = 0

    def check(name, cond, extra=""):
        nonlocal ok, total
        total += 1
        if cond:
            ok += 1
        log(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

    log_txt, good = run_one(c, "churn.csv", r"D:\ML_help\sample_data\客户流失预测.csv",
                            "是否流失", "classification",
                            ["lr", "knn", "nb", "dt", "rf", "et", "ada", "gbc",
                             "xgboost", "lightgbm", "catboost", "svm"],
                            id_cols=["客户ID"])
    log(log_txt)
    check("classification 12 models + shap", good)

    log_txt, good = run_one(c, "house.csv", r"D:\ML_help\sample_data\房价预测.csv",
                            "成交价", "regression",
                            ["lr", "knn", "dt", "rf", "et", "ada", "gbc",
                             "xgboost", "lightgbm", "catboost"],
                            id_cols=["房源ID"])
    log(log_txt)
    check("regression 10 models + shap", good)

    r = c.get(f"{BASE}/api/health")
    check("health after all", r.status_code == 200)

    log(f"\n==== {ok}/{total} checks passed ====")
    print(out.getvalue())
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()

