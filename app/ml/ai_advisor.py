# -*- coding: utf-8 -*-
"""
[可选增强] QClaw Agent AI 解读接口。
调用 QClaw 智能体对训练结果进行自动解读与调参建议，返回 Markdown 文本。

设计要点：
- 通过 OpenAI 兼容的 /v1/chat/completions 接口调用 QClaw Agent；
- 配置缺失或调用失败时优雅降级：返回基于规则生成的本地解读，
  保证前端功能可用（毕设答辩演示不依赖外部服务）；
- 不引入额外第三方库（用 urllib 标准库实现）。
"""
import json
import os
import urllib.request

# ---- 可配置项（建议放环境变量，避免硬编码密钥）----
QCLAW_API_BASE = os.environ.get("QCLAW_API_BASE", "http://127.0.0.1:8001/v1")
QCLAW_API_KEY = os.environ.get("QCLAW_API_KEY", "")
QCLAW_MODEL = os.environ.get("QCLAW_MODEL", "qclaw/pool-deepseek-v4-flash")


def ai_interpret(result: dict) -> str:
    """
    对训练结果做 AI 解读。result 为 /api/train 的返回 dict。
    优先调用 QClaw Agent；失败则回退本地规则解读。
    """
    try:
        return _call_qclaw(result)
    except Exception as e:
        return _local_fallback(result) + f"\n\n> （QClaw Agent 未连接，已使用本地规则解读：{e}）"


def _build_prompt(result: dict) -> str:
    """构造发送给 Agent 的提示词。"""
    task = "分类" if result["task_type"] == "classification" else "回归"
    model_lines = "\n".join(
        f"- {m['name']}: {m['metrics']}" for m in result["models"])
    best = result["best_model"]
    imp_lines = "\n".join(
        f"- {it['feature']}: {it['importance']}" for it in result["feature_importance"][:5])
    return f"""你是一名机器学习实验分析师。请根据以下训练结果，用中文给出简洁的解读（200字内）：
1. 实验结果解读（最优模型为什么好/差）
2. 数据与特征发现（特征重要性说明了什么）
3. 后续调参或改进建议（针对最优模型）

任务类型：{task}，目标列：{result['target_col']}，样本数：{result['n_samples']}
模型对比：
{model_lines}
最优模型：{best['name']}（{best['reason']}）指标 {best['metrics']}
特征重要性 Top5：
{imp_lines}"""


def _call_qclaw(result: dict) -> str:
    """调用 QClaw Agent（OpenAI 兼容协议）。"""
    payload = {
        "model": QCLAW_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个严谨的机器学习实验分析师，回答精炼、专业。"},
            {"role": "user", "content": _build_prompt(result)},
        ],
        "temperature": 0.3,
        "max_tokens": 600,
    }
    req = urllib.request.Request(
        f"{QCLAW_API_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {QCLAW_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def _local_fallback(result: dict) -> str:
    """本地规则解读（QClaw 不可用时兜底）。"""
    task = "分类" if result["task_type"] == "classification" else "回归"
    best = result["best_model"]
    top = result["feature_importance"][:3]
    top_str = "、".join(f"{t['feature']}" for t in top) if top else "（无）"
    return (f"**实验解读（本地规则）**\n\n"
            f"- 任务类型：{task}，样本 {result['n_samples']} 条，对比 {len(result['models'])} 个基础模型。\n"
            f"- 最优模型为 **{best['name']}**（{best['reason']}），关键指标：{best['metrics']}。\n"
            f"- 特征重要性显示 **{top_str}** 是影响预测的主要因素，可作为后续特征工程重点。\n"
            f"- 建议：对最优模型做 GridSearchCV 超参数搜索；若类别不均衡可尝试 SMOTE。")
