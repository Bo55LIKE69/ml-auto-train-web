/* ===== 标签选择页逻辑：加载概览 -> 配置任务 -> 触发训练 ===== */

const targetSelect = document.getElementById("targetSelect");
const taskType = document.getElementById("taskType");
const idColsBox = document.getElementById("idColsBox");
const warnBox = document.getElementById("warnBox");
const configForm = document.getElementById("configForm");
const trainBtn = document.getElementById("trainBtn");
const trainStatus = document.getElementById("trainStatus");
const modelPickField = document.getElementById("modelPickField");
const modelPickBox = document.getElementById("modelPickBox");

const fileId = new URLSearchParams(location.search).get("file_id");
let modelCatalog = { classification: [], regression: [] };

/* ---- 加载概览，填充下拉框 ---- */
async function loadOverview() {
  if (!fileId) {
    targetSelect.innerHTML = '<option value="">未提供 file_id，请先上传数据</option>';
    return;
  }
  try {
    const resp = await fetch(`/api/explore?file_id=${fileId}`);
    const info = await resp.json();
    if (!resp.ok) throw new Error(info.detail || "加载失败");

    // 目标列下拉：全部列，推荐项优先显示
    const suggested = info.suggested_target;
    const options = [...info.columns]
      .sort((a, b) => (b.name === suggested) - (a.name === suggested))
      .map(c => `<option value="${esc(c.name)}" ${c.name === suggested ? "selected" : ""}>
        ${esc(c.name)}（${c.dtype}，${c.unique} 取值）</option>`)
      .join("");
    targetSelect.innerHTML = options || '<option value="">无可用列</option>';

    // 疑似 ID 列复选框（默认勾选剔除）
    if (info.id_like_cols.length) {
      idColsBox.innerHTML = info.id_like_cols.map(c =>
        `<label><input type="checkbox" value="${esc(c)}" checked> ${esc(c)}</label>`).join("");
    } else {
      idColsBox.innerHTML = '<p class="hint">未检测到疑似 ID 列</p>';
    }

    // 风险提示（缺失严重 / 样本过少）
    const warns = [];
    if (info.n_samples < 100) warns.push(`[提示] 样本数仅 ${info.n_samples}，结果可靠性有限`);
    const heavy = info.columns.filter(c => c.missing_rate > 0.3);
    if (heavy.length) warns.push(`[提示] ${heavy.length} 列缺失率超过 30%（${heavy.map(c => c.name).join("、")}）`);
    if (warns.length) { warnBox.hidden = false; warnBox.textContent = warns.join("\n"); }
  } catch (err) {
    targetSelect.innerHTML = `<option value="">${esc(err.message)}</option>`;
  }
}

/* ---- 加载模型清单（/api/models），供手动模式勾选 ---- */
async function loadModels() {
  try {
    const resp = await fetch("/api/models");
    if (!resp.ok) return;
    modelCatalog = await resp.json();
  } catch (e) { /* 静默失败，手动模式会提示 */ }
}

/* ---- 模式切换：auto -> 隐藏模型勾选；manual -> 展示 ---- */
document.querySelectorAll('input[name="mode"]').forEach(r => {
  r.addEventListener("change", () => {
    const manual = document.querySelector('input[name="mode"]:checked').value === "manual";
    modelPickField.hidden = !manual;
    if (manual && modelCatalog.classification.length) renderModelPick();
  });
});

/* ---- 渲染模型勾选（按当前任务类型）---- */
function renderModelPick() {
  const type = taskType.value === "regression" ? "regression" : "classification";
  const models = modelCatalog[type];
  if (!models.length) {
    modelPickBox.innerHTML = '<p class="hint">模型清单加载失败，请刷新重试</p>';
    return;
  }
  modelPickBox.innerHTML = models.map(m =>
    `<label class="model-check"><input type="checkbox" value="${esc(m.key)}" checked>
      <span><strong>${esc(m.name)}</strong><small>${esc(m.desc || "")}</small></span></label>`
  ).join("");
}

/* ---- 提交训练 ---- */
configForm.addEventListener("submit", async e => {
  e.preventDefault();
  const target = targetSelect.value;
  if (!target) return showStatus("请先选择目标标签列", "error");

  trainBtn.disabled = true;
  trainBtn.innerHTML = '<span class="spinner"></span>训练中（多模型对比，约需几秒~几十秒）…';
  showStatus("正在执行：预处理 -> 多模型训练 -> 评估 -> 绘图 -> 生成报告…", "info");

  const idCols = [...document.querySelectorAll("#idColsBox input:checked")].map(i => i.value);
  const mode = document.querySelector('input[name="mode"]:checked').value;
  let modelSet = [];
  if (mode === "manual") {
    modelSet = [...document.querySelectorAll("#modelPickBox input:checked")].map(i => i.value);
    if (!modelSet.length) {
      showStatus("请至少勾选 1 个模型", "error");
      trainBtn.disabled = false;
      trainBtn.textContent = "开始训练";
      return;
    }
  }
  const body = { file_id: fileId, target_col: target, task_type: taskType.value, id_cols: idCols };
  if (mode === "manual") body.model_set = modelSet;
  try {
    const resp = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "训练失败");

    // 记录本次 task_id，结果页可自动加载
    localStorage.setItem("ml_last_task_id", data.task_id);
    location.href = `/result.html?task_id=${data.task_id}`;
  } catch (err) {
    showStatus("失败：" + err.message, "error");
    trainBtn.disabled = false;
    trainBtn.textContent = "开始训练";
  }
});

function showStatus(msg, type) {
  trainStatus.hidden = false;
  trainStatus.className = `status ${type}`;
  trainStatus.textContent = msg;
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

loadOverview();
loadModels();
