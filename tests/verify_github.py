# -*- coding: utf-8 -*-
"""验证 GitHub 仓库内容。"""
import subprocess
import sys

sys.path.insert(0, r"D:\ML_help\.venv\Lib\site-packages")
import httpx

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
h = {"Authorization": f"token {cred.get('password', '')}", "Accept": "application/vnd.github+json"}

r = httpx.get("https://api.github.com/repos/Bo55LIKE69/ml-auto-train-web", headers=h, timeout=30)
d = r.json()
print("REPO:", d["full_name"], "| private:", d["private"], "| default_branch:", d["default_branch"])

r2 = httpx.get("https://api.github.com/repos/Bo55LIKE69/ml-auto-train-web/contents/", headers=h, timeout=30)
if r2.status_code == 200:
    files = [f["name"] for f in r2.json()]
    print("TOP_LEVEL:", files)
else:
    print("CONTENTS_STATUS:", r2.status_code, r2.text[:200])

# 提交信息
r3 = httpx.get("https://api.github.com/repos/Bo55LIKE69/ml-auto-train-web/commits/main", headers=h, timeout=30)
if r3.status_code == 200:
    c = r3.json()
    print("LATEST_COMMIT:", c["sha"][:8], "|", c["commit"]["message"].splitlines()[0])
