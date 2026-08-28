# -*- coding: utf-8 -*-
"""验证 4 个页面的侧边栏结构完整性"""
import re
import httpx

pages = ["/", "/select.html", "/result.html", "/dashboard.html"]
c = httpx.Client(timeout=10)
ok = True
for p in pages:
    r = c.get(f"http://127.0.0.1:8000{p}")
    html = r.text
    checks = {
        "app-shell open": '<div class="app-shell">' in html,
        "sidebar": '<aside class="sidebar">' in html,
        "app-main": '<div class="app-main">' in html,
        "shell close": '</div><!-- /app-shell -->' in html,
        "brand": 'sidebar-brand-name' in html,
        "nav links": html.count('sidebar-link') >= 8,  # 4 links * (class + maybe active)
        "section labels": 'sidebar-section-label' in html,
        "footer": 'sidebar-footer' in html,
        "badge": 'sidebar-step-badge' in html,
    }
    # 统计 active 数量（应为 1）
    active = len(re.findall(r'sidebar-link active', html))
    checks["exactly 1 active"] = active == 1
    bad = [k for k, v in checks.items() if not v]
    status = "OK" if not bad else f"BAD: {bad}"
    if bad:
        ok = False
    print(f"{p}: {status} (active={active})")

# CSS 关键样式检查
css = c.get("http://127.0.0.1:8000/css/style.css").text
for sel in [".sidebar", ".sidebar-link", ".sidebar-link.active", ".app-main",
            ".sidebar-brand", ".sidebar-footer", ".sidebar-step-badge",
            ".sidebar-link-icon"]:
    if sel not in css:
        print(f"CSS MISSING: {sel}")
        ok = False
print("CSS sidebar selectors: all present" if ok else "CSS ISSUES FOUND")
print("ALL PASS" if ok else "FAIL")
