# -*- coding: utf-8 -*-
"""模块2 接口测试：上传 + 探查。
运行：cd D:/ML_help && .venv/Scripts/python -m pytest tests -v
"""
import io

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# 10 行小样本，包含 ID 列、数值列、类别列、中文列名
SAMPLE_CSV = """学号,年龄,成绩,专业,是否通过
S001,19,85,计算机,是
S002,20,78,计算机,是
S003,21,62,软件工程,否
S004,20,90,计算机,是
S005,19,55,软件工程,否
S006,22,72,计算机,是
S007,20,88,软件工程,是
S008,21,60,计算机,否
S009,19,95,计算机,是
S010,20,68,软件工程,是
"""


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_and_explore():
    # 1. 上传
    r = client.post(
        "/api/upload",
        files={"file": ("成绩表.csv", io.BytesIO(SAMPLE_CSV.encode("utf-8")), "text/csv")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    file_id = data["file_id"]
    assert data["filename"] == "成绩表.csv"
    assert data["size"] > 0

    # 2. 探查
    r = client.get("/api/explore", params={"file_id": file_id})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["n_samples"] == 10
    assert info["n_features"] == 5
    assert "学号" in info["id_like_cols"]          # ID 列被识别
    assert info["suggested_target"] == "是否通过"   # 候选标签建议
    assert len(info["preview"]) == 5               # 预览 5 行


def test_upload_bad_extension():
    r = client.post("/api/upload", files={"file": ("a.txt", io.BytesIO(b"x"), "text/plain")})
    assert r.status_code == 400


def test_explore_not_found():
    r = client.get("/api/explore", params={"file_id": "deadbeef"})
    assert r.status_code == 404
