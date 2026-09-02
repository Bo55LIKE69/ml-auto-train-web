# -*- coding: utf-8 -*-
"""
全局配置模块：统一管理文件存储路径、允许的上传类型、任务目录等。
所有路径默认位于 D:/ML_help 下，可通过环境变量 ML_HELP_HOME 覆盖（便于换机器）。
遵循规格书 §4「全局护栏配置」：所有资源限制集中在此处，禁止在业务代码中硬编码。
"""
import os
from pathlib import Path

# 项目根目录：默认 D:/ML_help（可通过环境变量 ML_HELP_HOME 修改）
BASE_DIR = Path(os.environ.get("ML_HELP_HOME", r"D:/ML_help")).resolve()

# 上传文件存储目录
UPLOAD_DIR = BASE_DIR / "uploads"
# 训练产物目录（图表/报告/复现脚本，按 task_id 子目录隔离）
OUTPUT_DIR = BASE_DIR / "outputs"
# 前端静态文件目录
STATIC_DIR = BASE_DIR / "app" / "static"

# ========== 文件限制（规格书 §4） ==========
MAX_FILE_SIZE_MB = 50                # 上传文件最大 50MB
MAX_UPLOAD_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_ROW_COUNT = 200_000              # 最大行数（规格书护栏）
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
ALLOWED_EXTENSIONS = SUPPORTED_EXTENSIONS
CSV_ENCODINGS_TO_TRY = ["utf-8", "gbk", "gb2312", "latin1"]   # 中文兼容
CSV_ENCODINGS = CSV_ENCODINGS_TO_TRY

# ========== 任务限制（规格书 §4） ==========
TRAIN_TIMEOUT_SECONDS = 600          # 单任务超时 10 分钟
MAX_CONCURRENT_TASKS = 2             # 最大并发任务数
DEFAULT_FOLD = 5                     # 默认交叉验证折数
RANDOM_STATE = 42                    # ★ 固定随机种子，保证可复现性
SESSION_ID = RANDOM_STATE

# 训练集划分比例（保留 train_test_split 用，CV 场景下为 holdout 比例）
TEST_SIZE = 0.3

# ========== 模型集合（规格书 §4 附录 A，12 模型） ==========
# 默认完整模型集合（compare_models include 列表）
DEFAULT_MODEL_SET = [
    "lr",        # Logistic Regression
    "knn",       # K Neighbors
    "nb",        # Naive Bayes
    "dt",        # Decision Tree
    "rf",        # Random Forest
    "et",        # Extra Trees
    "ada",       # AdaBoost
    "gbc",       # Gradient Boosting
    "xgboost",   # XGBoost
    "lightgbm",  # LightGBM
    "catboost",  # CatBoost
    "svm",       # SVM（线性核，避免慢）
]
# 超时后第一次降级：减少模型数量
FALLBACK_MODEL_SET_SMALL = ["lr", "knn", "dt", "rf", "et", "gbc", "xgboost", "lightgbm"]
# 第二次降级：只用最快的前三个
FALLBACK_MODEL_SET_MINIMAL = ["lr", "dt", "rf"]

# ========== 交叉验证（v1.2.0：真正执行 K 折 CV，不再只是记账字段）==========
# CV_ENABLED：默认开启。毕设论文中「K 折交叉验证」为方法章节标配。
CV_ENABLED = True
# CV_MAX_SAMPLES：样本数超过此值时自动跳过 CV（12 模型 × K 折在大样本上极慢）
CV_MAX_SAMPLES = 20_000
# CV_MIN_SAMPLES：样本数少于此值时不建议 CV（每折样本太少，方差大）
CV_MIN_SAMPLES = 20

# ========== SVM 性能护栏 ==========
# 样本数超过此值时 SVM 自动切线性核（RBF 在大样本上复杂度过高）
SVM_LINEAR_THRESHOLD = 3_000

# ========== 存储路径 ==========
STORAGE_ROOT = OUTPUT_DIR
DEMO_DATA_DIR = BASE_DIR / "demo-data"

# ========== 任务类型自动判断阈值（规格书 §4） ==========
CLASSIFICATION_UNIQUE_THRESHOLD = 20   # 唯一值 ≤ 此数 → 可能是分类标签
ID_COLUMN_RATIO_THRESHOLD = 0.95       # 唯一值占比 ≥ 此数 → 疑似 ID 列

# ========== LLM 配置（AI 解读用） ==========
LLM_PROVIDER = os.environ.get("QCLAW_API_BASE", "") or "openai"
LLM_MODEL = os.environ.get("QCLAW_API_KEY", "") or "gpt-4o-mini"
LLM_MAX_TOKENS = 2000

# ========== 报告配置（规格书 §4） ==========
REPORT_CHART_DPI = 150                # 报告内嵌图分辨率（打印级）
REPORT_DISCLAIMER = (
    "以下内容由 AutoML 毕设实验工作台自动生成，"
    "仅供学习参考使用，请核对后引用至论文或报告中。"
)

# ========== 可解释性与图表增强（v1.0.0） ==========
SHAP_MAX_SAMPLES = 100               # SHAP 可解释性采样上限（控制耗时）
CORRELATION_MAX_FEATURES = 30        # 相关性热力图最大特征数（超出取方差 Top N）
PDF_ENABLED = True                   # 是否启用 PDF 报告导出（需本机安装 LibreOffice）
PDF_TIMEOUT_SECONDS = 120            # LibreOffice 转换超时

# ========== 特征工程选项（v1.1.0，默认 auto 保持原行为） ==========
# 缺失值策略：auto(中位数/众数) / median / most_frequent / constant
IMPUTE_STRATEGY_CHOICES = ("auto", "median", "most_frequent", "constant")
# 特征缩放：auto(标准化) / standard / minmax / none
SCALER_CHOICES = ("auto", "standard", "minmax", "none")
# 类别编码：auto(独热) / onehot / label
CAT_ENCODING_CHOICES = ("auto", "onehot", "label")
# 常数填充值（strategy=constant 时使用）
IMPUTE_CONSTANT_VALUE = 0

# 特征工程选项默认值（全 auto = 完全等价于旧行为）
DEFAULT_FE_OPTS = {
    "impute_strategy": "auto",
    "scaler": "auto",
    "cat_encoding": "auto",
}

# LibreOffice 可执行文件路径（自动探测）
SOFFICE_CANDIDATES = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]


# ========== 可选依赖清单（v1.2.0：依赖自检用，缺失时在界面明确提示）==========
# key -> (pip 包名, import 模块名, 用途说明, 缺失后影响)
OPTIONAL_DEPS = {
    "xgboost":    ("xgboost", "xgboost", "XGBoost 模型", "12 模型降级为 11 个"),
    "lightgbm":   ("lightgbm", "lightgbm", "LightGBM 模型", "12 模型降级为 11 个"),
    "catboost":   ("catboost", "catboost", "CatBoost 模型", "12 模型降级为 11 个"),
    "shap":       ("shap", "shap", "SHAP 可解释性分析", "不生成 shap_summary.png"),
    "python-docx": ("python-docx", "docx", "Word 实验报告", "不生成 report.docx / report.pdf"),
    "joblib":     ("joblib", "joblib", "模型持久化", "无法保存模型，在线预测不可用"),
}


def check_optional_deps() -> dict:
    """
    检查可选依赖是否可用。
    返回 {key: {"available": bool, "purpose": str, "impact": str, "error": str|None}}
    仅在需要时调用（如 /api/deps 或训练前），不阻塞启动。
    """
    import importlib
    out = {}
    for key, (pip_name, mod_name, purpose, impact) in OPTIONAL_DEPS.items():
        try:
            importlib.import_module(mod_name)
            out[key] = {"available": True, "purpose": purpose,
                        "impact": None, "error": None, "pip_name": pip_name}
        except Exception as e:
            out[key] = {"available": False, "purpose": purpose,
                        "impact": impact, "error": str(e), "pip_name": pip_name}
    return out


def ensure_dirs():
    """确保运行时目录存在（幂等，启动时调用）。"""
    for d in (UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR):
        d.mkdir(parents=True, exist_ok=True)


# LibreOffice 路径探测（在 ensure_dirs 调用前不能依赖 Path 对象，用 shutil）
import shutil as _shutil
SOFFICE_EXE = None
for _p in SOFFICE_CANDIDATES:
    if Path(_p).exists():
        SOFFICE_EXE = _p
        break
if SOFFICE_EXE is None:
    _found = _shutil.which("soffice") or _shutil.which("soffice.exe")
    if _found:
        SOFFICE_EXE = _found

print(f"[config] LibreOffice PDF conversion: {'ENABLED at ' + SOFFICE_EXE if SOFFICE_EXE else 'DISABLED (soffice not found)'}")

