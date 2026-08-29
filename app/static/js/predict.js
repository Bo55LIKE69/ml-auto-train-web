/* ===== 在线预测页逻辑：加载任务 -> 上传新数据 -> 预测 -> 预览/下载 ===== */

const taskIdEl = document.getElementById("taskId");
const loadTaskBtn = document.getElementById("loadTaskBtn");
const taskInfo = document.getElementById("taskInfo");
const predDropZone = document.getElementById("predDropZone");
const predFileBtn = document.getElementById("predFileBtn");
const predFile = document.getElementById("predFile");
const predFileCard = document.getElementById("predFileCard");
const predFileName = document.getElementById("predFileName");
const predFileSize = document.getElementById("predFileSize");
const predFileRemove = document.getElementById("predFileRemove");
const predictBtn = document.getElementById("predictBtn");
const predictStatus = document.getElementById("predictStatus");
const resultCard = document.getElementById("resultCard");
const predStats = document.getElementById("predStats");
const predTable = document.getElementById("predTable");
const dlPred = document.getElementById("dlPred");

let taskMeta = null;
let selectedFile = null;

/* ---- 初始化：URL ?task_id 优先 ---- */
(function init() {
  const id = new URLSearchParams(location.search).get("task_id");
  if (id) {
    taskIdEl.value = id;
    loadTask();
  }
})();

loadTaskBtn.addEventListener("click", loadTask);

async function loadTask() {
  const id = taskIdEl.value.trim();
  if (!id) return showStatus("请先输入任务 ID", "error");
  taskInfo.hidden = true;
  loadTaskBtn.disabled = true;
  loadTaskBtn.textContent = "加载中…";
  try {
    const resp = await fetch(`/api/result/${id}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "任务不存在");
    taskMeta = data;
    taskInfo.hidden = false;
    taskInfo.innerHTML =
      `已加载任务 <b class="mono">${esc(id)}</b>：最优模型 <b>${esc(data.best_model.name)}</b>`
      + ` · ${data.task_type === "classification" ? "分类" : "回归"}`
      + ` · 特征 ${data.n_features_after_prep} 维`
      + (data.tuned && data.tuned.enabled ? " · 已超参调优" : "");
    showStatus("任务信息已加载，请上传新数据", "info");
    refreshPredictBtn();
  } catch (err) {
    showStatus("加载失败：" + err.message, "error");
  } finally {
    loadTaskBtn.disabled = false;
    loadTaskBtn.textContent = "加载任务信息";
  }
}

/* ---- 文件选择 ---- */
predFileBtn.addEventListener("click", () => predFile.click());
predFile.addEventListener("change", () => {
  if (predFile.files[0]) setFile(predFile.files[0]);
});
predFileRemove.addEventListener("click", () => {
  selectedFile = null;
  predFile.value = "";
  predFileCard.hidden = true;
  predDropZone.style.display = "";
  refreshPredictBtn();
});
["dragover", "dragenter"].forEach(ev =>
  predDropZone.addEventListener(ev, e => { e.preventDefault(); predDropZone.classList.add("dragover"); }));
["dragleave", "drop"].forEach(ev =>
  predDropZone.addEventListener(ev, e => { e.preventDefault(); predDropZone.classList.remove("dragover"); }));
predDropZone.addEventListener("drop", e => {
  const f = e.dataTransfer.files[0];
  if (f) { predFile.files = e.dataTransfer.files; setFile(f); }
});

function setFile(f) {
  if (!/\.(csv|xlsx|xls)$/i.test(f.name)) {
    return showStatus("不支持的文件格式，请选择 CSV / XLSX / XLS", "error");
  }
  selectedFile = f;
  predFileName.textContent = f.name;
  predFileSize.textContent = fmtSize(f.size);
  predFileCard.hidden = false;
  predDropZone.style.display = "none";
  showStatus("文件已选择，点击运行预测", "info");
  refreshPredictBtn();
}

function refreshPredictBtn() {
  predictBtn.disabled = !(taskMeta && selectedFile);
}

/* ---- 运行预测 ---- */
predictBtn.addEventListener("click", async () => {
  if (!taskMeta || !selectedFile) return;
  predictBtn.disabled = true;
  predictBtn.innerHTML = '<span class="spinner"></span>预测中…';
  showStatus("正在加载模型并预测…", "info");
  try {
    const fd = new FormData();
    fd.append("file", selectedFile);
    const resp = await fetch(`/api/predict/${taskMeta.task_id}`, { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "预测失败");
    renderResult(data);
    showStatus(`预测完成，共 ${data.n_predicted} 行`, "success");
  } catch (err) {
    showStatus("失败：" + err.message, "error");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "运行预测";
  }
});

function renderResult(data) {
  predStats.innerHTML = [
    { num: data.model_name, lbl: "预测模型" },
    { num: data.n_predicted, lbl: "预测行数" },
    { num: data.columns.length, lbl: "输出列数" },
  ].map(s => `<div class="stat-box"><div class="num">${esc(String(s.num))}</div><div class="lbl">${s.lbl}</div></div>`).join("");
  const keys = data.columns;
  predTable.querySelector("thead").innerHTML =
    `<tr>${keys.map(k => `<th>${esc(k)}</th>`).join("")}</tr>`;
  predTable.querySelector("tbody").innerHTML = data.preview.map(row =>
    `<tr>${keys.map(k => `<td>${esc(row[k] ?? "")}</td>`).join("")}</tr>`).join("")
    || `<tr><td colspan="${keys.length}">（无预览）</td></tr>`;
  dlPred.href = data.download_url;
  dlPred.setAttribute("download", `${data.task_id}_predictions.csv`);
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth" });
}

/* ---- 工具 ---- */
function showStatus(msg, type) {
  predictStatus.hidden = false;
  predictStatus.className = `status ${type}`;
  predictStatus.textContent = msg;
}
function fmtSize(b) {
  return b > 1048576 ? (b / 1048576).toFixed(1) + " MB" : (b / 1024).toFixed(1) + " KB";
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
