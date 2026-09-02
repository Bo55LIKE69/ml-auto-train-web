# 📊 表格机器学习自动训练工具（Web版）

> **作者**：谢泓铎 · GIS实验室  
> 面向**本科毕设学生**的表格数据机器学习自动训练 Web 工具：
上传 CSV / Excel 数据集，选择目标标签，系统自动完成
**数据探查 → 预处理 → 多模型对比训练 → 评估 → 可视化 → 生成 Markdown 实验报告**，
并可下载**可复现的 Python 训练脚本**。

> 全部模型训练逻辑为手写 scikit-learn 代码，**未使用任何成熟 AutoML 库**，便于在论文中阐述原理。

---

## ✨ 功能特性

| 环节 | 说明 |
|------|------|
| 📤 数据上传 | 支持 CSV / Excel（xlsx、xls），≤50MB，自动识别 utf-8 / gbk / gb18030 编码 |
| 🔍 数据探查 | 样本数、特征数、缺失统计、列类型推断、疑似 ID 列检测、前 5 行预览、自动推荐目标标签 |
| 🧹 自动预处理 | 缺失值填充、类别特征编码（独热/标签）、数值标准化、ID 列剔除 |
| 🤖 多模型训练 | 分类：逻辑回归 / 随机森林 / 决策树 / SVM；回归：线性回归 / Ridge / 决策树 / 随机森林 |
| 📈 自动评估 | 分类：Accuracy / Precision / Recall / F1 / AUC + 混淆矩阵；回归：R² / MAE / RMSE + 散点图 |
| 🖼️ 可视化 | 混淆矩阵、真实值 vs 预测值、特征重要性 Top15 |
| 📄 报告生成 | 自动生成 Markdown 实验报告（含模型对比表、指标解读、复现方法） |
| ⬇️ 结果下载 | Markdown 报告 / 可复现 Python 脚本 / 结果 JSON |
| 🤖 AI 解读（可选） | 调用 QClaw Agent 自动解读实验结果并给出调参建议（未配置时降级为本地规则解读） |

## 📝 最近更新

- **训练稳健性增强**：逐模型异常隔离（单模型失败不再拖垮整任务）；稀有类别自动降级分层抽样；实现**真 5 折交叉验证**（均值±标准差，样本 > 2 万自动跳过），并据此选最优模型；特征重要性改用最优模型来源；超时两级降级（small / minimal 模型集）并修复补跑去重 bug；任务状态落盘持久化（服务重启可恢复）。
- **依赖自检**：新增 `/api/deps` 接口，前端侧边栏实时显示 XGBoost / LightGBM / CatBoost / SHAP / Word 报告 / 模型持久化 可用性；`requirements.txt` 补全此前缺失的 6 个包（xgboost / lightgbm / catboost / shap / python-docx / joblib）。
- **UI 升级**：抽离共享侧边栏（五页导航一致 + 运行环境面板）；修复嵌套 `<a>` 非法结构；实验结果页新增模型对比横向柱状图（最优高亮 + 动画）；补全预测页缺失主题样式。
- 当前共支持 **12 个模型**（分类 / 回归），全部为手写 scikit-learn / XGBoost / LightGBM / CatBoost 代码，便于在毕设论文中阐述原理。

## 🚀 快速开始（Windows）

```bat
:: 1. 首次安装（自动创建虚拟环境并安装依赖）
setup.bat

:: 2. 启动服务
start.bat
```

浏览器访问 **http://127.0.0.1:8000**

### 手动方式

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Linux / macOS

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🧑‍💻 使用流程

1. **上传数据** → 页面上查看数据概览（缺失/类型/预览）
2. **选择标签** → 选目标列、任务类型（自动判断/分类/回归）、勾选剔除 ID 列
3. **自动训练** → 等待多模型对比完成（几秒~几十秒）
4. **查看结果** → 模型对比表、图表、特征重要性，下载报告与复现脚本

## 🔌 API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/upload` | 上传数据，返回 file_id |
| GET | `/api/explore?file_id=` | 数据概览与推荐标签 |
| POST | `/api/train` | 提交训练任务 |
| GET | `/api/download/{task_id}/result.json` | 下载结果 JSON |
| GET | `/api/download/{task_id}/report.md` | 下载实验报告 |
| GET | `/api/download/{task_id}/script.py` | 下载复现脚本 |
| GET | `/api/ai/interpret/{task_id}` | AI 解读（可选增强） |

## 🗂️ 目录结构

```
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置
│   ├── routers/             # upload / explore / train / download / ai
│   ├── ml/                  # 数据探查、预处理、训练、评估、绘图、报告、AI解读
│   └── static/              # 前端页面（原生 HTML+JS）
├── sample_data/             # 示例数据集（学生成绩）
├── tests/                   # 单元测试 + 端到端测试
├── requirements.txt
├── setup.bat / start.bat    # Windows 一键脚本
└── 部署文档.md
```

## 🧪 测试

```bash
.venv\Scripts\python -m pytest tests -v          # 单元测试
.venv\Scripts\python tests\test_http_e2e.py      # HTTP 端到端（需服务已启动）
.venv\Scripts\python tests\test_e2e_pipeline.py  # 训练流水线端到端
```

## 🛠️ 技术栈

FastAPI · pandas · scikit-learn · matplotlib · 原生 HTML/JS

## 📄 许可

MIT

## 👤 作者

**谢泓铎** · GIS实验室

- GitHub：[@Bo55LIKE69](https://github.com/Bo55LIKE69)
- 项目仓库：<https://github.com/Bo55LIKE69/ml-auto-train-web>
