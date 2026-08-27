/* ===== 上传页逻辑：文件上传 + 数据概览展示 ===== */

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const fileHint = document.getElementById("fileHint");
const uploadForm = document.getElementById("uploadForm");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const overviewCard = document.getElementById("overviewCard");

let selectedFile = null;

/* ---- 拖拽/点击选择文件 ---- */
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files[0] || null;
  fileHint.textContent = selectedFile ? `已选择：${selectedFile.name}（${fmtSize(selectedFile.size)}）` : "未选择文件";
});
["dragover", "dragenter"].forEach(ev =>
  dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add("dragover"); }));
["dragleave", "drop"].forEach(ev =>
  dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove("dragover"); }));
dropZone.addEventListener("drop", e => {
  const f = e.dataTransfer.files[0];
  if (f) { selectedFile = f; fileInput.files = e.dataTransfer.files; fileHint.textContent = `已选择：${f.name}（${fmtSize(f.size)}）`; }
});

/* ---- 上传 + 探查 ---- */
uploadForm.addEventListener("submit", async e => {
  e.preventDefault();
  if (!selectedFile) return showStatus("请先选择文件", "error");

  const fd = new FormData();
  fd.append("file", selectedFile);
  uploadBtn.disabled = true;
  uploadBtn.innerHTML = '<span class="spinner"></span>上传中…';
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
    showStatus("数据读取成功，可进行下一步", "info");
  } catch (err) {
    showStatus("失败：" + err.message, "error");
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "上传并探查数据";
  }
});

/* ---- 渲染概览 ---- */
function renderOverview(fileId, info) {
  // 统计卡片
  overviewStats.innerHTML = [
    { num: info.n_samples, lbl: "样本数" },
    { num: info.n_features, lbl: "特征数" },
    { num: info.missing_total, lbl: "缺失值总数" },
    { num: info.numeric_cols.length, lbl: "数值列" },
    { num: info.categorical_cols.length, lbl: "类别列" },
    { num: info.id_like_cols.length, lbl: "疑似ID列" },
  ].map(s => `<div class="stat-box"><div class="num">${s.num}</div><div class="lbl">${s.lbl}</div></div>`).join("");

  // 列信息表
  colTable.querySelector("tbody").innerHTML = info.columns.map(c => `
    <tr>
      <td>${esc(c.name)}</td>
      <td>${c.dtype}</td>
      <td>${c.missing}</td>
      <td>${(c.missing_rate * 100).toFixed(1)}%</td>
      <td>${c.unique}</td>
      <td>${c.is_id_like ? "是(疑似ID)" : "否"}</td>
    </tr>`).join("");

  // 预览表（动态表头）
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
  uploadStatus.hidden = false;
  uploadStatus.className = `status ${type}`;
  uploadStatus.textContent = msg;
}
function fmtSize(b) { return b > 1048576 ? (b / 1048576).toFixed(1) + " MB" : (b / 1024).toFixed(1) + " KB"; }
function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
