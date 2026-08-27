# -*- coding: utf-8 -*-
"""扫描前端文件中的 emoji 字符，输出文件:行:内容。"""
import re
import pathlib
import sys

ROOT = pathlib.Path(r"D:\ML_help\app\static")

# emoji 检测（含变体选择符、旗帜、符号类）
emoji_re = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # 杂项符号和象形文字
    "\U00002600-\U000027BF"   # 杂项符号(含⚠✅❌⭐)
    "\U00002B00-\U00002BFF"   # 杂项符号和箭头
    "\U0000FE0F"              # 变体选择符
    "\U00002190-\U000021FF"   # 箭头(←↑→↓ 等)
    "\U00002300-\U000023FF"   # 杂项技术符号
    "\U00002B50"              # ⭐
    "\U00002764"              # ❤
    "\U00002705"              # ✅
    "\U0000274C"              # ❌
    "]+"
)

for p in sorted(ROOT.rglob("*")):
    if not p.is_file() or p.suffix not in (".html", ".js", ".css"):
        continue
    text = p.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in emoji_re.finditer(line):
            chars = "".join(f"U+{ord(c):04X}" for c in m.group())
            print(f"{p.relative_to(ROOT)}:{lineno}: [{chars}] {line.strip()[:80]}", file=sys.stderr)
