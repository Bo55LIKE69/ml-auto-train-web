# -*- coding: utf-8 -*-
"""老链路快速回归（针对 8010 新代码）"""
import sys
import time

import httpx

BASE = "http://127.0.0.1:8010"
fails = []


def check(n, c, e=""):
    print(("PASS" if c else "FAIL"), n, e)
    if not c:
        fails.append(n)


with open("sample_data/学生成绩分类.csv", "rb") as f:
    fid = httpx.post(f"{BASE}/api/upload",
                     files={"file": ("学生成绩分类.csv", f, "text/csv")}).json().get("file_id")
check("upload", bool(fid))
r = httpx.post(f"{BASE}/api/train", json={
    "file_id": fid, "target_col": "是否通过",
    "task_type": "classification", "id_cols": ["学号"]})
tid = r.json().get("task_id")
check("submit", bool(tid))
st, cur = "", 0
for _ in range(120):
    time.sleep(2)
    d = httpx.get(f"{BASE}/api/tasks/{tid}", params={"log_cursor": cur}).json()
    st = d.get("status")
    cur = d.get("log_cursor", cur)
    if st in ("completed", "failed"):
        break
check("completed", st == "completed", f"st={st}")
res = httpx.get(f"{BASE}/api/result/{tid}").json()
check("fe_opts in result", res.get("fe_opts", {}).get("impute_strategy") == "auto",
      str(res.get("fe_opts")))
check("12 models", len(res.get("models", [])) == 12, f"n={len(res.get('models', []))}")
print("FAILS:", fails if fails else "NONE")
sys.exit(1 if fails else 0)
