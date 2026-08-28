# -*- coding: utf-8 -*-
import httpx

BASE = "http://127.0.0.1:8000"
checks = {
    "/": ["class=\"stepper\"", "feature-list", "class=\"file-card\"",
          "DATASET / STEP 01", "dz-icon", "eyebrow"],
    "/select.html": ["CONFIG / STEP 02", "class=\"stepper\"", "mode-switch"],
    "/result.html": ["RESULT / STEP 03", "class=\"stepper\""],
}
for page, toks in checks.items():
    html = httpx.get(BASE + page).text
    miss = [t for t in toks if t not in html]
    print(f"{page}: {'OK' if not miss else 'MISSING ' + str(miss)}")
