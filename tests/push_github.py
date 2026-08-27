# -*- coding: utf-8 -*-
"""用凭据管理器 token 推送到 GitHub（避免交互弹窗卡死）。"""
import subprocess
import sys

sys.path.insert(0, r"D:\ML_help\.venv\Lib\site-packages")
import httpx

# 1. 取 token
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
user = cred.get("username", "Bo55LIKE69")
token = cred.get("password", "")
if not token:
    print("ERROR: 未获取到 token")
    sys.exit(1)

# 2. 用带 token 的 URL 推送（token 只在本进程内使用）
remote = f"https://{user}:{token}@github.com/{user}/ml-auto-train-web.git"
r = subprocess.run(
    ["git", "push", "-u", "origin", "main"],
    cwd=r"D:\ML_help",
    env={**__import__("os").environ, "GIT_ASKPASS": "echo", "GIT_TERMINAL_PROMPT": "0"},
    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
)
print("RC:", r.returncode)
print("STDOUT:", r.stdout[-2000:])
print("STDERR:", r.stderr[-2000:])

# 3. 更新 remote URL 为纯净地址
subprocess.run(["git", "remote", "set-url", "origin", f"https://github.com/{user}/ml-auto-train-web.git"], cwd=r"D:\ML_help")
print("remote reset to clean URL")
