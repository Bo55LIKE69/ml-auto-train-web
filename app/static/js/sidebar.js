/* ===== 运行环境面板：填充每个页面侧边栏里的 <div id="envList"> =====
   拉取后端 /api/deps，把模型/报告依赖的可用性显式展示出来，
   学生一眼就能看到 XGBoost / SHAP / Word 报告 等是否可用。
   侧边栏导航本身仍是静态 HTML（避免注入闪烁），本脚本只负责这个面板。
*/
(function () {
  const box = document.getElementById("envList");
  if (!box) return;

  function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  fetch("/api/deps")
    .then(r => (r.ok ? r.json() : null))
    .then(d => {
      if (!d) { box.innerHTML = '<span class="env-loading">状态不可用</span>'; return; }
      const deps = d.dependencies || {};
      const labelMap = {
        xgboost: "XGBoost", lightgbm: "LightGBM", catboost: "CatBoost",
        shap: "SHAP 可解释", "python-docx": "Word 报告", joblib: "模型持久化",
      };
      const rows = Object.keys(deps).map(k => {
        const ok = deps[k].available;
        const name = labelMap[k] || k;
        const tip = ok ? "" : ` title="${esc(deps[k].impact || "依赖缺失")}"`;
        return `<div class="env-item"${tip}><span class="env-dot ${ok ? "ok" : "bad"}"></span>${name}${ok ? "" : " · 缺失"}</div>`;
      }).join("");
      const summary = d.all_ready
        ? `<div class="env-summary ok"><span class="env-dot ok"></span>全功能就绪</div>`
        : `<div class="env-summary bad"><span class="env-dot bad"></span>部分功能缺失</div>`;
      box.innerHTML = summary + rows;
    })
    .catch(() => { box.innerHTML = '<span class="env-loading">状态不可用</span>'; });
})();
