# -*- coding: utf-8 -*-
"""HTTP 端到端验证 v2：上传 -> 探查 -> 异步训练 -> 轮询 -> 下载 Word/图表/ZIP。
手动运行：python tests/test_http_e2e_v2.py
pytest 收集安全：所有执行代码都在 main() 内，不污染 pytest 捕获。
"""
import io
import json
import sys
import time

BASE = "http://127.0.0.1:8000"
fails = []


def check(name, cond, extra=""):
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name} {extra}")
    if not cond:
        fails.append(name)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    import httpx

    # 1. health
    r = httpx.get(f"{BASE}/api/health")
    check("health", r.status_code == 200, f"({r.status_code})")

    # 2. upload 学生成绩示例.csv
    with open(r"D:\ML_help\sample_data\学生成绩示例.csv", "rb") as f:
        r = httpx.post(f"{BASE}/api/upload", files={"file": ("学生成绩示例.csv", f, "text/csv")})
    check("upload", r.status_code == 200, f"({r.status_code})")
    data = r.json()
    file_id = data.get("file_id")
    check("upload file_id", bool(file_id), f"file_id={file_id}")
    print("  upload resp:", json.dumps(data, ensure_ascii=False)[:150])

    # 3. explore
    r = httpx.get(f"{BASE}/api/explore", params={"file_id": file_id})
    check("explore", r.status_code == 200, f"({r.status_code})")
    info = r.json()
    check("explore suggested_target", info.get("suggested_target") == "是否通过",
          f"suggested={info.get('suggested_target')}")

    # 4. 异步训练（用 9 个模型避免 CatBoost 首次编译慢）
    r = httpx.post(f"{BASE}/api/train", json={
        "file_id": file_id, "target_col": "是否通过", "task_type": "classification",
        "id_cols": ["学号"],
        "model_set": ["lr", "knn", "nb", "dt", "rf", "et", "ada", "gbc", "svm"],
    })
    check("train submit", r.status_code == 200, f"({r.status_code})")
    td = r.json()
    task_id = td.get("task_id")
    check("train task_id", bool(task_id), f"task_id={task_id}")
    print("  train resp:", json.dumps(td, ensure_ascii=False)[:150])

    # 5. 轮询直到完成
    status, lines = "training", []
    t0 = time.time()
    while status == "training" and time.time() - t0 < 180:
        time.sleep(2)
        r = httpx.get(f"{BASE}/api/tasks/{task_id}", params={"log_cursor": len(lines)})
        d = r.json()
        lines.extend(d.get("log_lines", []))
        status = d.get("status", "training")
    check("train completed", status == "completed", f"status={status}, 耗时{time.time()-t0:.0f}s")
    print(f"  日志行数: {len(lines)}")
    print("  日志最后 3 行:")
    for l in lines[-3:]:
        print("   ", l)
    if status == "completed":
        check("metrics_summary", d.get("result_available") is True,
              f"best={d.get('metrics_summary', {}).get('best_model')}")

    # 6. result.json 完整结果
    r = httpx.get(f"{BASE}/api/result/{task_id}")
    check("result.json", r.status_code == 200)
    rj = r.json()
    check("result models>=9", len(rj.get("models", [])) >= 9, f"({len(rj.get('models', []))} models)")
    check("result best", rj["best_model"]["name"] == "逻辑回归", f"best={rj['best_model']['name']}")

    # 7. 下载 Word 报告
    r = httpx.get(f"{BASE}/api/download/{task_id}/report.docx")
    check("download docx", r.status_code == 200 and r.content[:2] == b"PK",
          f"({r.status_code}, {len(r.content)} bytes, magic={r.content[:2]})")

    # 8. 下载图表 ZIP
    r = httpx.get(f"{BASE}/api/download/{task_id}/charts")
    check("download charts zip", r.status_code == 200 and r.content[:2] == b"PK",
          f"({r.status_code}, {len(r.content)} bytes)")

    # 9. 下载全部产物 ZIP
    r = httpx.get(f"{BASE}/api/download/{task_id}/all")
    check("download all zip", r.status_code == 200 and r.content[:2] == b"PK",
          f"({r.status_code}, {len(r.content)} bytes)")

    # 10. pipeline_ir
    r = httpx.get(f"{BASE}/api/download/{task_id}/pipeline_ir.json")
    check("pipeline_ir", r.status_code == 200)
    ir = r.json()
    check("ir has setup_config", "setup_config" in ir and ir["setup_config"]["fold"] == 5)

    # 11. 训练日志
    r = httpx.get(f"{BASE}/api/download/{task_id}/training.log")
    check("training.log", r.status_code == 200 and len(r.text) > 100, f"({len(r.text)} chars)")

    print(f"\n=== 结果: {len(fails)} 失败 / 11 项 ===")
    if fails:
        print("失败项:", fails)
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
