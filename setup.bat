@echo off
REM ============================================================
REM  表格ML自动训练工具 - 首次安装脚本 (Windows)
REM  创建虚拟环境 + 安装依赖
REM ============================================================
chcp 65001 >nul
cd /d %~dp0

echo [1/3] 创建虚拟环境...
python -m venv .venv
if errorlevel 1 (
    echo [错误] 创建虚拟环境失败，请确认已安装 Python 3.10+
    pause
    exit /b 1
)

echo [2/3] 升级 pip...
call .venv\Scripts\python -m pip install --upgrade pip

echo [3/3] 安装依赖...
call .venv\Scripts\python -m pip install -r requirements.txt

echo.
echo 安装完成！运行 start.bat 启动服务，浏览器访问 http://127.0.0.1:8000
pause
