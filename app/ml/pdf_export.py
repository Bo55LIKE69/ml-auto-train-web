# -*- coding: utf-8 -*-
"""
PDF 导出模块（v1.0.0 新增）：
使用本机 LibreOffice 将 Word 报告（.docx）转换为 PDF。
依赖：LibreOffice soffice.exe（config.py 探测路径）。

用法（独立）：
    from app.ml.pdf_export import convert_docx_to_pdf
    ok = convert_docx_to_pdf("input.docx", "output.pdf")
"""
import subprocess
import time
from pathlib import Path

from app.config import PDF_TIMEOUT_SECONDS, SOFFICE_EXE

# 探测 soffice 失败时的友好提示
_NO_SOFFICE_MSG = (
    "LibreOffice 未安装或不在默认路径，无法转换 PDF。"
    "请访问 https://www.libreoffice.org/download/download/ 下载安装。"
)


def convert_docx_to_pdf(docx_path: Path | str,
                        pdf_path: Path | str,
                        timeout: int = None) -> bool:
    """
    使用 LibreOffice headless 将 docx 转换为 pdf。

    参数：
        docx_path  Word 文件路径（.docx）
        pdf_path   输出 PDF 路径
        timeout    超时秒数（默认 PDF_TIMEOUT_SECONDS）

    返回：
        True 转换成功，False 失败
    """
    soffice = SOFFICE_EXE
    if soffice is None:
        print("[pdf_export] LibreOffice not found. " + _NO_SOFFICE_MSG)
        return False

    docx_path = Path(docx_path).resolve()
    pdf_path = Path(pdf_path).resolve()

    if not docx_path.is_file():
        print(f"[pdf_export] Source file not found: {docx_path}")
        return False

    # 输出目录必须存在
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    timeout = timeout or PDF_TIMEOUT_SECONDS
    out_dir = str(pdf_path.parent)

    try:
        # LibreOffice headless 转换：--headless --convert-to pdf --outdir <dir> <file>
        # 注意：soffice 在 Windows 上有时需要用 "soffice.com" 而不是 "soffice.exe"
        # 我们先试 soffice.exe，失败后再试 soffice.com
        for binary in ("soffice.exe", "soffice.com"):
            exe = str(Path(soffice).parent / binary) if soffice else binary
            try:
                proc = subprocess.run(
                    [exe,
                     "--headless",
                     "--convert-to", "pdf",
                     "--outdir", out_dir,
                     str(docx_path)],
                    capture_output=True,
                    timeout=timeout,
                )
                break
            except FileNotFoundError:
                continue

        # 检查是否生成了 PDF（LibreOffice 输出文件名与 docx 相同）
        if pdf_path.exists():
            # 如果同名文件已存在（上次转换残留），检查大小合理性
            if pdf_path.stat().st_size < 1024:
                print("[pdf_export] PDF file too small, conversion may have failed.")
                return False
            print(f"[pdf_export] PDF generated: {pdf_path}")
            return True

        # LibreOffice 有时把 PDF 输出到同目录下
        same_dir_pdf = docx_path.with_suffix(".pdf")
        if same_dir_pdf.exists() and same_dir_pdf != pdf_path:
            same_dir_pdf.rename(pdf_path)
            return True

        print(f"[pdf_export] Conversion failed. stdout: {proc.stdout}")
        print(f"[pdf_export] stderr: {proc.stderr}")
        return False

    except subprocess.TimeoutExpired:
        print(f"[pdf_export] LibreOffice conversion timeout ({timeout}s)")
        return False
    except Exception as e:
        print(f"[pdf_export] Error: {e}")
        return False
