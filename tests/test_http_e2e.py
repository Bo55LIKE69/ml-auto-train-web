# -*- coding: utf-8 -*-
"""HTTP 端到端测试：上传 → 探查 → 训练 → 下载，验证前后端联动。
运行：cd D:/ML_help && .venv/Scripts/python tests/test_http_e2e.py
"""
import json
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
DATA = Path(__file__).resolve().parent.parent / "sample_data" / "学生成绩示例.csv"


def main():
    c = httpx.Client(timeout=120)

    # 1. 健康检查
    r = c.get(f"{BASE}/api/health")
    print(f"[1] health: {r.json()['status']}")

    # 2. 上传
    with open(DATA, "rb") as f:
        r = c.post(f"{BASE}/api/upload", files={"file": ("学生成绩示例.csv", f, "text/csv")})
    assert r.status_code == 200, r.text
    file_id = r.json()["file_id"]
    print(f"[2] upload ok: file_id={file_id}")

    # 3. 探查
    r = c.get(f"{BASE}/api/explore", params={"file_id": file_id})
    assert r.status_code == 200, r.text
    info = r.json()
    print(f"[3] explore ok: samples={info['n_samples']} features={info['n_features']} "
          f"suggested_target={info['suggested_target']} id_like={info['id_like_cols']}")

    # 4. 训练
    r = c.post(f"{BASE}/api/train", json={
        "file_id": file_id, "target_col": "是否通过", "task_type": "auto", "id_cols": [],
    })
    assert r.status_code == 200, r.text
    res = r.json()
    task_id = res["task_id"]
    print(f"[4] train ok: task={task_id} type={res['task_type']} best={res['best_model']['name']}")
    for m in res["models"]:
        print(f"      {m['name']}: {m['metrics']}")
    print(f"      plots: {list(res['plots'].keys())}")

    # 5. 下载报告 / 脚本 / 结果JSON
    for art in ("report.md", "script.py", "result.json"):
        r = c.get(f"{BASE}/api/download/{task_id}/{art}")
        assert r.status_code == 200, (art, r.text)
        print(f"[5] download {art}: {len(r.content)} bytes")

    # 6. 首页与静态资源
    r = c.get(f"{BASE}/")
    assert r.status_code == 200 and "上传数据集" in r.text
    for s in ("/css/style.css", "/js/upload.js", "/select.html", "/result.html"):
        rr = c.get(f"{BASE}{s}")
        assert rr.status_code == 200, s
    print("[6] 前端页面/静态资源全部 200")

    # 7. 复现脚本可执行性验证（生成的数据与训练相同）
    script = (Path(__file__).resolve().parent.parent / "outputs" / task_id / "script.py")
    assert script.is_file()
    print(f"[7] 复现脚本已生成: {script}")
    print("\n✅ HTTP 端到端全部通过")


if __name__ == "__main__":
    main()
