/* ===== 上传页逻辑：文件选择（File Card 状态）/ 拖拽 / 上传 + 数据概览 ===== */

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const fileCard = document.getElementById("fileCard");
const fileNameEl = document.getElementById("fileName");
const fileSizeEl = document.getElementById("fileSize");
const fileRemove = document.getElementById("fileRemove");
const uploadForm = document.getElementById("uploadForm");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const overviewCard = document.getElementById("overviewCard");
const overviewStats = document.getElementById("overviewStats");
const colTable = document.getElementById("colTable");
const previewTable = document.getElementById("previewTable");
const goSelectBtn = document.getElementById("goSelectBtn");

const MAX_SIZE = 50 * 1024 * 1024; // 50MB
let selectedFile = null;

/* ---- 选择文件后切换到 File Card 状态 ---- */
function setFile(f) {
  if (!f) return;
  // 扩展名校验（与后端一致）
  if (!/\.(csv|xlsx|xls)$/i.test(f.name)) {
    return showStatus("不支持的文件格式，请选择 CSV / XLSX / XLS", "error");
  }
  if (f.size > MAX_SIZE) {
    return showStatus(`文件超过 50MB（当前 ${(f.size / 1048576).toFixed(1)} MB）`, "error");
  }
  selectedFile = f;
  fileNameEl.textContent = f.name;
  fileSizeEl.textContent = fmtSize(f.size);
  fileCard.hidden = false;
  dropZone.style.display = "none";
  uploadBtn.disabled = false;
  uploadBtn.textContent = "上传并探索数据";
}

/* ---- 点击/拖拽选择 ---- */
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});
fileRemove.addEventListener("click", (e) => {
  e.stopPropagation();
  selectedFile = null;
  fileInput.value = "";
  fileCard.hidden = true;
  dropZone.style.display = "";
  showStatus("", "info");
});

["dragover", "dragenter"].forEach(ev =>
  dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add("dragover"); }));
["dragleave", "drop"].forEach(ev =>
  dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove("dragover"); }));
dropZone.addEventListener("drop", e => {
  const f = e.dataTransfer.files[0];
  if (f) { fileInput.files = e.dataTransfer.files; setFile(f); }
});

/* ---- 上传 + 探查 ---- */
uploadForm.addEventListener("submit", async e => {
  e.preventDefault();
  if (!selectedFile) return showStatus("请先选择文件", "error");

  const fd = new FormData();
  fd.append("file", selectedFile);
  uploadBtn.disabled = true;
  uploadBtn.innerHTML = '<span class="spinner"></span>正在上传…';
  showStatus("上传中，请稍候…", "info");

  try {
    const resp = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "上传失败");

    showStatus("上传成功，正在读取数据概览…", "info");
    const exp = await fetch(`/api/explore?file_id=${data.file_id}`);
    const info = await exp.json();
    if (!exp.ok) throw new Error(info.detail || "数据读取失败");

    renderOverview(data.file_id, info);
    showStatus("数据集上传成功，可进行下一步", "success");
  } catch (err) {
    showStatus("失败：" + err.message, "error");
    uploadBtn.disabled = false;
    uploadBtn.textContent = "上传并探索数据";
  }
});

/* ---- 渲染概览 ---- */
function renderOverview(fileId, info) {
  overviewStats.innerHTML = [
    { num: info.n_samples, lbl: "样本数" },
    { num: info.n_features, lbl: "特征数" },
    { num: info.missing_total, lbl: "缺失值总数" },
    { num: info.numeric_cols.length, lbl: "数值列" },
    { num: info.categorical_cols.length, lbl: "类别列" },
    { num: info.id_like_cols.length, lbl: "疑似ID列" },
  ].map(s => `<div class="stat-box"><div class="num">${s.num}</div><div class="lbl">${s.lbl}</div></div>`).join("");

  colTable.querySelector("tbody").innerHTML = info.columns.map(c => `
    <tr>
      <td>${esc(c.name)}</td>
      <td>${c.dtype}</td>
      <td>${c.missing}</td>
      <td>${(c.missing_rate * 100).toFixed(1)}%</td>
      <td>${c.unique}</td>
      <td>${c.is_id_like ? "是(疑似ID)" : "否"}</td>
    </tr>`).join("");

  const keys = info.preview.length ? Object.keys(info.preview[0]) : [];
  previewTable.querySelector("thead").innerHTML =
    `<tr>${keys.map(k => `<th>${esc(k)}</th>`).join("")}</tr>`;
  previewTable.querySelector("tbody").innerHTML = info.preview.map(row =>
    `<tr>${keys.map(k => `<td>${esc(row[k] ?? "")}</td>`).join("")}</tr>`).join("");

  goSelectBtn.href = `/select.html?file_id=${fileId}`;
  overviewCard.hidden = false;
  overviewCard.scrollIntoView({ behavior: "smooth" });
}

/* ---- 工具函数 ---- */
function showStatus(msg, type) {
  if (!msg) { uploadStatus.hidden = true; uploadStatus.className = "status"; return; }
  uploadStatus.hidden = false;
  uploadStatus.className = `status ${type}`;
  uploadStatus.textContent = msg;
}
function fmtSize(b) {
  return b > 1048576 ? (b / 1048576).toFixed(1) + " MB" : (b / 1024).toFixed(1) + " KB";
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
