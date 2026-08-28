# -*- coding: utf-8 -*-
"""E2E 验证：特征工程选项（fe_opts）走 HTTP 全链路。
覆盖：
  A. 分类 + fe_opts={minmax, label}（自定义特征工程）
  B. 回归 + 默认 auto（回归兼容性）
  C. 校验 result.fe_opts / IR setup_config / 复现脚本 FE_OPTS 一致性
"""
import json
import sys
import time

import httpx

BASE = "http://127.0.0.1:8010"
fails = []


def check(name, cond, extra=""):
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name} {extra}")
    if not cond:
        fails.append(name)


def wait_task(task_id, timeout=300):
    lines, t0 = [], time.time()
    while time.time() - t0 < timeout:
        time.sleep(2)
        r = httpx.get(f"{BASE}/api/tasks/{task_id}", params={"log_cursor": len(lines)})
        d = r.json()
        lines.extend(d.get("log_lines", []))
        if d.get("status") in ("completed", "failed"):
            return d
    return {"status": "timeout", "log_lines": lines}


def upload(path, fname):
    with open(path, "rb") as f:
        r = httpx.post(f"{BASE}/api/upload", files={"file": (fname, f, "text/csv")})
    d = r.json()
    return d.get("file_id"), d


# ---- A. 分类 + 自定义特征工程（minmax 缩放 + label 编码）----
print("=== A. 分类 + fe_opts(minmax, label) ===")
fid, _ = upload("sample_data/客户流失预测.csv", "客户流失预测.csv")
check("A upload", bool(fid))
r = httpx.post(f"{BASE}/api/train", json={
    "file_id": fid,
    "target_col": "是否流失",
    "task_type": "classification",
    "id_cols": ["客户ID"],
    "fe_opts": {"scaler": "minmax", "cat_encoding": "label"},
})
d = r.json()
check("A train submit", r.status_code == 200, f"status={r.status_code}")
task_id = d.get("task_id")
done = wait_task(task_id)
check("A completed", done.get("status") == "completed", f"status={done.get('status')}")

res = httpx.get(f"{BASE}/api/result/{task_id}").json()
fe = res.get("fe_opts", {})
check("A result.fe_opts", fe.get("scaler") == "minmax" and fe.get("cat_encoding") == "label",
      f"fe_opts={fe}")

# IR setup_config 一致性
ir = json.loads(httpx.get(f"{BASE}/api/download/{task_id}/pipeline_ir.json").content)
sc = ir.get("setup_config", {})
check("A IR setup_config", sc.get("scaler") == "minmax" and sc.get("cat_encoding") == "label",
      f"setup_config.fe={sc.get('scaler')}/{sc.get('cat_encoding')}")

# 复现脚本 FE_OPTS
script = httpx.get(f"{BASE}/api/download/{task_id}/script.py").text
check("A script FE_OPTS", '"scaler": "minmax"' in script and '"cat_encoding": "label"' in script)
check("A script import OrdinalEncoder", "OrdinalEncoder" in script)
check("A script import MinMaxScaler", "MinMaxScaler" in script)

# ---- B. 回归 + 默认 auto（回归链路兼容）----
print("=== B. 回归 + fe_opts 默认(auto) ===")
fid2, _ = upload("sample_data/房价预测.csv", "房价预测.csv")
check("B upload", bool(fid2))
r = httpx.post(f"{BASE}/api/train", json={
    "file_id": fid2,
    "target_col": "成交价",
    "task_type": "regression",
    "id_cols": ["房产ID"],
})
d = r.json()
check("B train submit", r.status_code == 200)
done = wait_task(d["task_id"])
check("B completed", done.get("status") == "completed", f"status={done.get('status')}")
res2 = httpx.get(f"{BASE}/api/result/{d['task_id']}").json()
fe2 = res2.get("fe_opts", {})
check("B fe_opts auto", fe2.get("impute_strategy") == "auto" and fe2.get("scaler") == "auto",
      f"fe_opts={fe2}")

# ---- C. 非法值回退 auto ----
print("=== C. 非法 fe_opts 回退 ===")
r = httpx.post(f"{BASE}/api/train", json={
    "file_id": fid,
    "target_col": "是否流失",
    "task_type": "classification",
    "fe_opts": {"scaler": "bogus", "cat_encoding": "nope"},
})
check("C train submit", r.status_code == 200)
done = wait_task(r.json()["task_id"])
check("C completed", done.get("status") == "completed")
res3 = httpx.get(f"{BASE}/api/result/{r.json()['task_id']}").json()
fe3 = res3.get("fe_opts", {})
check("C fallback auto", fe3.get("scaler") == "auto" and fe3.get("cat_encoding") == "auto",
      f"fe_opts={fe3}")

print()
print(f"=== 结果: {len(fails)} 失败 ===")
if fails:
    print("失败项:", fails)
    sys.exit(1)
print("全部通过")
