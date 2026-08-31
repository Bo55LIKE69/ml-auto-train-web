# -*- coding: utf-8 -*-
"""为已发布的 GitHub release 追加作者署名（谢泓铎 · GIS实验室）。

用法：
  python scripts/create_release.py           # 更新 v1.0.0 与 v1.1.0 两个 release 的署名
  python scripts/create_release.py v1.1.0    # 只更新指定 tag
"""
import base64
import json
import subprocess
import sys
import urllib.request

REPO = "Bo55LIKE69/ml-auto-train-web"
AUTHOR_BLOCK = (
    "\n\n---\n\n"
    "**作者**：谢泓铎 · GIS实验室  \n"
    "GitHub：[@Bo55LIKE69](https://github.com/Bo55LIKE69)  \n"
    "项目仓库：<https://github.com/Bo55LIKE69/ml-auto-train-web>"
)

DEFAULT_TAGS = ["v1.0.0", "v1.1.0"]


def get_token():
    payload = (
        "protocol=https\n"
        "host=github.com\n"
        "path=Bo55LIKE69/ml-auto-train-web.git\n"
        "\n"
    )
    p = subprocess.run(
        ["git", "credential", "fill"],
        input=payload, capture_output=True, text=True, encoding="utf-8",
    )
    for line in p.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("未能从凭据管理器获取 token")


def api(method, url, token, data=None):
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "ml-auto-train-release")
    if data is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode("utf-8")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read().decode("utf-8", "ignore")


def list_releases(token):
    url = f"https://api.github.com/repos/{REPO}/releases?per_page=100"
    status, body = api("GET", url, token)
    if status != 200:
        raise RuntimeError(f"列出 release 失败: {status} {body[:300]}")
    return json.loads(body)


V100_NOTES = """## v1.0.0 发布说明

首个正式版本。面向本科毕设学生的表格数据机器学习自动训练 Web 工具：

- 上传 CSV / Excel → 选目标标签 → 自动数据探查 / 预处理 / 多模型对比训练 / 评估 / 可视化 / 生成 Markdown 实验报告
- 12 个分类模型 + 10 个回归模型（手写 scikit-learn，未用成熟 AutoML 库，便于论文阐述原理）
- 5 折交叉验证、超时降级、训练日志捕获、pipeline_ir 记账
- 自动 / 手动双模式选模型、训练进度轮询
- Word 七章实验报告 + 可复现 Python 训练脚本下载
- 相关性热力图、SHAP  beeswarm 图、PDF 导出、历史任务页（实验工作台）
"""


def create_release(token, tag, notes):
    url = f"https://api.github.com/repos/{REPO}/releases"
    data = {
        "tag_name": tag,
        "name": tag,
        "body": notes + AUTHOR_BLOCK,
        "draft": False,
        "prerelease": False,
    }
    status, body = api("POST", url, token, data)
    if status in (200, 201):
        j = json.loads(body)
        print(f"[OK] 创建 release {tag} -> {j.get('html_url')}")
    else:
        print(f"[失败] 创建 {tag}: {status} {body[:400]}")


def main():
    token = get_token()
    print("token 获取成功，长度:", len(token))
    targets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TAGS
    releases = {r["tag_name"]: r for r in list_releases(token)}
    for tag in targets:
        if tag in releases:
            rel = releases[tag]
            old_body = rel.get("body") or ""
            if "谢泓铎" in old_body:
                print(f"[跳过] {tag} 已含作者署名")
                continue
            new_body = old_body + AUTHOR_BLOCK
            rid = rel["id"]
            url = f"https://api.github.com/repos/{REPO}/releases/{rid}"
            status, body = api("PATCH", url, token, {"body": new_body})
            if status in (200, 201):
                print(f"[OK] {tag} 已追加作者署名 -> {rel.get('html_url')}")
            else:
                print(f"[失败] {tag}: {status} {body[:400]}")
        else:
            notes = V100_NOTES if tag == "v1.0.0" else ""
            print(f"[创建] {tag} 无 release 页，拟创建")
            create_release(token, tag, notes)


if __name__ == "__main__":
    main()
