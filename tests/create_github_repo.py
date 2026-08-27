# -*- coding: utf-8 -*-
"""通过 Windows 凭据管理器获取 GitHub 凭据，用 API 创建仓库。"""
import json
import subprocess
import sys

try:
    import httpx
except ImportError:
    sys.path.insert(0, r"D:\ML_help\.venv\Lib\site-packages")
    import httpx

# 1. 从 git credential manager 取凭据
proc = subprocess.run(
    ["git", "credential", "fill"],
    input="protocol=https\nhost=github.com\n\n",
    capture_output=True, text=True, encoding="utf-8", timeout=30,
)
cred = {}
for line in proc.stdout.splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        cred[k] = v
print("username:", cred.get("username", "?"))
token = cred.get("password", "")
if not token:
    print("ERROR: 未获取到 token")
    sys.exit(1)

# 2. 创建仓库
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github+json",
}
payload = {
    "name": "ml-auto-train-web",
    "description": "Web table ML auto-training tool: upload CSV/Excel, auto train multi-models, generate Markdown report (FastAPI + sklearn)",
    "private": False,
    "auto_init": False,
}
r = httpx.post("https://api.github.com/user/repos", json=payload, headers=headers, timeout=60)
print("API_STATUS:", r.status_code)
if r.status_code in (200, 201):
    print("CREATED:", r.json()["html_url"])
else:
    print("RESP:", r.text[:500])
