@echo off
REM ============================================================
REM  表格ML自动训练工具 - Windows 一键启动脚本
REM  首次运行请先执行 setup.bat 安装依赖
REM ============================================================
chcp 65001 >nul
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
    echo [错误] 未找到虚拟环境 .venv，请先运行 setup.bat
    pause
    exit /b 1
)

echo 正在启动服务: http://127.0.0.1:8000
call .venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
