# -*- coding: utf-8 -*-
"""验证服务返回的静态资源内容是否完整。"""
import re
import urllib.request

BASE = "http://127.0.0.1:8000"

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        return r.read().decode("utf-8", errors="replace")

for path in ["/", "/css/style.css", "/js/upload.js", "/js/select.js", "/js/result.js", "/select.html", "/result.html"]:
    body = get(path)
    print(f"{path}: {len(body)} chars")

# 检查 CSS 是否完整（应该包含设计系统关键标记）
css = get("/css/style.css")
checks = ["--accent", "logo-mark", "plot-card", "stats-grid", "prefers-reduced-motion"]
for c in checks:
    print(f"css contains '{c}':", c in css)

# 检查 emoji
emoji_re = re.compile("[\u2600-\u27BF\u2B00-\u2BFF\u1F300-\u1FAFF\uFE0F]")
for path in ["/", "/select.html", "/result.html", "/css/style.css", "/js/upload.js", "/js/select.js", "/js/result.js"]:
    body = get(path)
    found = emoji_re.findall(body)
    print(f"emoji in {path}: {found if found else 'none'}")
