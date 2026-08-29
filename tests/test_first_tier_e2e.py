# -*- coding: utf-8 -*-
"""第一梯队功能端到端验证：模型下载+在线预测 / 超参调优 / 评估图补全。
- 分类训练(tune=true) -> 校验 model_artifacts.joblib 存在 + predict 接口 + 评估图
- 回归训练 -> 校验 residual.png
用法: python tests/test_first_tier_e2e.py
"""
import io, json, os, time, uuid
import urllib.request, urllib.error
import pandas as pd

BASE = "http://127.0.0.1:8010"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(ROOT, "sample_data")
OUT = os.path.join(ROOT, "outputs")

def req(method, url, data=None, headers=None, raw=False):
    body = None
    if data is not None:
        if isinstance(data, (bytes, bytearray)):
            body = data
        else:
            body = json.dumps(data).encode("utf-8")
            headers = dict(headers or {}); headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        resp = urllib.request.urlopen(r, timeout=120)
        return resp.status, resp.read() if raw else resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")

def upload(path):
    name = os.path.basename(path)
    with open(path, "rb") as f:
        payload = f.read()
    boundary = "----qclawtest" + uuid.uuid4().hex[:8]
    form = (b"--" + boundary.encode() + b"\r\n"
            + b'Content-Disposition: form-data; name="file"; filename="' + name.encode() + b'"\r\n'
            + b"Content-Type: application/octet-stream\r\n\r\n" + payload + b"\r\n"
            + b"--" + boundary.encode() + b"--\r\n")
    h = {"Content-Type": "multipart/form-data; boundary=" + boundary}
    st, b = req("POST", BASE + "/api/upload", data=form, headers=h)
    assert st == 200, f"upload {name} {st}: {b[:200]}"
    return json.loads(b)["file_id"]

def wait_task(tid, timeout=200):
    start = time.time()
    while time.time() - start < timeout:
        st, b = req("GET", f"{BASE}/api/tasks/{tid}")
        d = json.loads(b)
        if d.get("status") in ("completed", "failed"):
            return d
        time.sleep(2)
    raise TimeoutError(f"task {tid} timeout")

def check_classification():
    print("[A] 分类训练 + 在线预测 + 评估图")
    fid = upload(os.path.join(SAMPLE, "客户流失预测.csv"))
    body = {"file_id": fid, "target_col": "是否流失", "task_type": "classification",
            "id_cols": [], "fe_opts": {}, "tune": True, "tune_budget": 12}
    st, b = req("POST", BASE + "/api/train", data=body)
    assert st == 200, f"train {st}: {b[:200]}"
    tid = json.loads(b)["task_id"]
    d = wait_task(tid)
    assert d["status"] == "completed", f"task failed: {d.get('error')}"
    # result.json
    st, b = req("GET", f"{BASE}/api/result/{tid}")
    res = json.loads(b)
    # 评估图
    for key in ("learning_curve", "roc_curve"):
        assert key in res["plots"], f"missing plot {key}: {list(res['plots'])}"
    # 模型文件
    mpath = os.path.join(OUT, tid, "model_artifacts.joblib")
    assert os.path.isfile(mpath), "model_artifacts.joblib not persisted"
    # 调优信息
    assert res.get("tuned", {}).get("enabled"), "tune not enabled in result"
    print(f"    best={res['best_model']['name']} tuned={res['tuned']}")
    # predict 接口：用原始文件当新数据（去掉目标列也可）
    name = os.path.basename(os.path.join(SAMPLE, "客户流失预测.csv"))
    with open(os.path.join(SAMPLE, "客户流失预测.csv"), "rb") as f:
        payload = f.read()
    boundary = "----qclawpred" + uuid.uuid4().hex[:8]
    form = (b"--" + boundary.encode() + b"\r\n"
            + b'Content-Disposition: form-data; name="file"; filename="' + name.encode() + b'"\r\n'
            + b"Content-Type: application/octet-stream\r\n\r\n" + payload + b"\r\n"
            + b"--" + boundary.encode() + b"--\r\n")
    h = {"Content-Type": "multipart/form-data; boundary=" + boundary}
    st, b = req("POST", f"{BASE}/api/predict/{tid}", data=form, headers=h)
    assert st == 200, f"predict {st}: {b[:200]}"
    pred = json.loads(b)
    assert pred["download_url"].endswith("predictions.csv")
    assert pred["n_predicted"] > 0
    assert os.path.isfile(os.path.join(OUT, tid, "predictions.csv"))
    print(f"    predict OK: {pred['n_predicted']} rows, model={pred['model_name']}")

def check_regression():
    print("[B] 回归训练 + 残差图")
    fid = upload(os.path.join(SAMPLE, "房价预测.csv"))
    body = {"file_id": fid, "target_col": "成交价", "task_type": "regression",
            "id_cols": [], "tune": False}
    st, b = req("POST", BASE + "/api/train", data=body)
    assert st == 200, f"train {st}: {b[:200]}"
    tid = json.loads(b)["task_id"]
    d = wait_task(tid)
    assert d["status"] == "completed", f"task failed: {d.get('error')}"
    st, b = req("GET", f"{BASE}/api/result/{tid}")
    res = json.loads(b)
    assert "residual" in res["plots"], f"missing residual: {list(res['plots'])}"
    assert "learning_curve" in res["plots"], f"missing learning_curve: {list(res['plots'])}"
    print(f"    best={res['best_model']['name']} plots={list(res['plots'])}")

def check_frontend():
    print("[C] 前端页面可达")
    for p in ("/predict.html", "/select.html", "/result.html", "/dashboard.html"):
        st, b = req("GET", BASE + p)
        assert st == 200 and "sidebar" in b, f"{p} {st}"
    # predict.js 存在
    st, b = req("GET", BASE + "/js/predict.js")
    assert st == 200, f"predict.js {st}"
    print("    all pages + predict.js OK")

if __name__ == "__main__":
    check_classification()
    check_regression()
    check_frontend()
    print("\nALL PASS")
