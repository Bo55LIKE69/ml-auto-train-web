/* ===== 结果展示页逻辑：训练轮询 -> 加载结果 -> 渲染模型对比/图表/下载 ===== */

const resultCard = document.getElementById("resultCard");
const emptyCard = document.getElementById("emptyCard");
const trainCard = document.getElementById("trainCard");
const warnBox = document.getElementById("warnBox");
const taskIdInput = document.getElementById("taskIdInput");
const loadForm = document.getElementById("loadForm");
const lastTaskLink = document.getElementById("lastTaskLink");

let current = null;
let pollTimer = null;
let logCursor = 0;

/* ---- 初始化：URL 参数优先，其次 localStorage ---- */
(function init() {
  const fromUrl = new URLSearchParams(location.search).get("task_id");
  const last = localStorage.getItem("ml_last_task_id");
  if (fromUrl) { checkTaskState(fromUrl); return; }
  if (last) {
    lastTaskLink.href = `/result.html?task_id=${last}`;
    lastTaskLink.textContent = `加载最近一次结果（${last}）`;
    lastTaskLink.parentElement.style.display = "";
    checkTaskState(last);
  }
})();

loadForm.addEventListener("submit", e => {
  e.preventDefault();
  const id = taskIdInput.value.trim();
  if (id) {
    history.replaceState(null, "", `/result.html?task_id=${id}`);
    checkTaskState(id);
  }
});

/* ---- 任务状态检查：训练中则轮询，完成则加载结果 ---- */
async function checkTaskState(taskId) {
  emptyCard.hidden = true;
  resultCard.hidden = true;
  trainCard.hidden = true;
  try {
    const resp = await fetch(`/api/tasks/${taskId}?log_cursor=0`);
    if (resp.status === 404) {
      // 任务不在内存中，可能是已完成的历史任务
      await loadResult(taskId);
      return;
    }
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "查询失败");

    if (data.status === "training") {
      showTrainCard(taskId, data);
      return;
    }
    if (data.status === "completed") {
      await loadResult(taskId);
      return;
    }
    if (data.status === "failed") {
      emptyCard.hidden = false;
      const errBox = document.createElement("div");
      errBox.className = "warn";
      errBox.textContent = "训练失败：" + (data.error?.message || "未知错误");
      emptyCard.prepend(errBox);
      return;
    }
    await loadResult(taskId);
  } catch (err) {
    emptyCard.hidden = false;
    alert("查询失败：" + err.message);
  }
}

/* ---- 训练进度卡片 + 2s 轮询 ---- */
function showTrainCard(taskId, data) {
  trainCard.hidden = false;
  trainTaskBadge.textContent = taskId;
  const logViewer = document.getElementById("logViewer");
  const bar = document.getElementById("trainProgressBar");
  const text = document.getElementById("trainProgressText");
  logViewer.textContent = (data.log_lines || []).join("\n") + "\n";
  logCursor = data.next_cursor || 0;

  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const resp = await fetch(`/api/tasks/${taskId}?log_cursor=${logCursor}`);
      const d = await resp.json();
      if (d.log_lines && d.log_lines.length) {
        logViewer.textContent += d.log_lines.join("\n") + "\n";
        logViewer.scrollTop = logViewer.scrollHeight;
        logCursor = d.next_cursor || logCursor;
      }
      // 进度估算：从日志中提取 "模型 x/N"
      const m = logViewer.textContent.match(/模型 (\d+)\/(\d+)/g);
      let pct = 5;
      if (m && m.length) {
        const last = m[m.length - 1].match(/(\d+)\/(\d+)/);
        pct = 10 + Math.round((last[1] / last[2]) * 70);
      }
      if (/全部完成|Word 报告/.test(logViewer.textContent)) pct = 100;
      bar.style.width = pct + "%";
      text.textContent = `训练进度 ${pct}%（2 秒自动刷新）`;

      if (d.status === "completed") {
        clearInterval(pollTimer);
        await loadResult(taskId);
      } else if (d.status === "failed") {
        clearInterval(pollTimer);
        emptyCard.hidden = false;
        const errBox = document.createElement("div");
        errBox.className = "warn";
        errBox.textContent = "训练失败：" + (d.error?.message || "未知错误");
        emptyCard.prepend(errBox);
      }
    } catch (err) {
      clearInterval(pollTimer);
      alert("轮询失败：" + err.message);
    }
  }, 2000);
}

/* ---- 加载结果 ---- */
async function loadResult(taskId) {
  if (pollTimer) clearInterval(pollTimer);
  emptyCard.hidden = true;
  resultCard.hidden = true;
  trainCard.hidden = true;
  showLoadingBar(taskId);
  try {
    const resp = await fetch(`/api/download/${taskId}/result.json`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "结果不存在");
    current = data;
    render(data);
  } catch (err) {
    hideLoadingBar();
    emptyCard.hidden = false;
    alert("加载失败：" + err.message);
  }
}

/* ---- 渲染 ---- */
function render(r) {
  // 头部徽章
  taskTypeBadge.textContent = r.task_type === "classification" ? "分类任务" : "回归任务";
  taskIdBadge.textContent = r.task_id;

  // 风险提示
  if (r.warnings && r.warnings.length) {
    warnBox.hidden = false;
    warnBox.textContent = r.warnings.join("\n");
  }

  // 数据集信息
  dataStats.innerHTML = [
    { num: r.n_samples, lbl: "样本数" },
    { num: r.n_features_raw, lbl: "原始特征数" },
    { num: r.n_features_after_prep, lbl: "预处理后维度" },
    { num: r.models.length, lbl: "对比模型数" },
    { num: r.class_names ? r.class_names.length : "—", lbl: r.class_names ? "类别数" : "连续值" },
    { num: r.drop_cols.length, lbl: "剔除列数" },
  ].map(s => `<div class="stat-box"><div class="num">${s.num}</div><div class="lbl">${s.lbl}</div></div>`).join("");

  // 模型对比表（动态表头）
  const headers = Object.keys(r.models[0].metrics);
  modelHead.innerHTML = `<th>模型</th>${headers.map(h => `<th>${h}</th>`).join("")}`;
  modelBody.innerHTML = r.models.map(m => `
    <tr class="${m.name === r.best_model.name ? "best-row" : ""}">
      <td>${esc(m.name)}${m.name === r.best_model.name ? " [最优]" : ""}</td>
      ${headers.map(h => `<td>${m.metrics[h] ?? "—"}</td>`).join("")}
    </tr>`).join("");

  // 最优模型框
  const bm = r.best_model;
  const metricStr = Object.entries(bm.metrics).filter(([, v]) => v !== null)
    .map(([k, v]) => `${k}=${v}`).join("，");
  bestBox.innerHTML = `<b>最优模型：${esc(bm.name)}</b>（${esc(bm.reason)}）<br>
    测试集指标：${metricStr}`;

  // 可视化
  const plotTitles = {
    confusion_matrix: "混淆矩阵（最优模型 · 测试集）",
    scatter: "真实值 vs 预测值（最优模型 · 测试集）",
    feature_importance: "特征重要性 Top15（随机森林）",
    metrics_comparison: "模型指标对比（测试集）",
    correlation: "特征相关性热力图（数据探索）",
    shap_summary: "SHAP 特征重要性（可解释性）",
  };
  plotArea.innerHTML = Object.entries(r.plots).map(([k, url]) => `
    <div class="plot-card">
      <img src="${url}" alt="${plotTitles[k] || k}" loading="lazy">
      <p>${plotTitles[k] || k}</p>
    </div>`).join("") || "<p class='hint'>（无可用图表）</p>";

  // AI 解读（可选增强，失败不阻塞）
  const aiBox = document.getElementById("aiBox");
  aiBox.hidden = true;
  fetch(`/api/ai/interpret/${r.task_id}`)
    .then(resp => resp.ok ? resp.json() : Promise.reject(new Error("AI 解读不可用")))
    .then(d => {
      aiBox.querySelector(".ai-content").innerHTML =
        "<div style='white-space:pre-wrap'>" + esc(d.interpretation) + "</div>";
      aiBox.hidden = false;
    })
    .catch(() => {});

  // 特征重要性表
  impTable.querySelector("tbody").innerHTML = (r.feature_importance || []).map((it, i) =>
    `<tr><td>${i + 1}</td><td>${esc(it.feature)}</td><td>${it.importance}</td></tr>`).join("")
    || "<tr><td colspan='3'>（无）</td></tr>";

  // 下载链接
  dlDocx.href = `/api/download/${r.task_id}/report.docx`;
  dlReport.href = `/api/download/${r.task_id}/report.md`;
  dlScript.href = `/api/download/${r.task_id}/script.py`;
  dlJson.href = `/api/download/${r.task_id}/result.json`;
  dlCharts.href = `/api/download/${r.task_id}/charts`;
  dlAll.href = `/api/download/${r.task_id}/all`;
  // PDF：触发后端 LibreOffice 转换（若不存在则现转）
  dlPdf.href = `/api/download/${r.task_id}/report.pdf`;

  hideLoadingBar();
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth" });
}

/* ---- 加载提示条 ---- */
function showLoadingBar(taskId) {
  let bar = document.getElementById("loadingBar");
  if (!bar) {
    bar = document.createElement("div");
    bar.id = "loadingBar";
    bar.className = "status info";
    bar.style.marginBottom = "16px";
    document.querySelector("main.container").prepend(bar);
  }
  bar.innerHTML = '<span class="spinner"></span>正在加载训练结果…';
}
function hideLoadingBar() {
  const bar = document.getElementById("loadingBar");
  if (bar) bar.remove();
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
