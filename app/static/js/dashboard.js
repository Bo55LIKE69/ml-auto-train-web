/* ===== 实验工作台：历史任务列表 ===== */

const tbody = document.querySelector("#taskTable tbody");
const countEl = document.getElementById("taskCount");
const emptyHint = document.getElementById("emptyHint");

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  const p = n => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
function fmtScore(key, v) {
  if (v === null || v === undefined) return "—";
  return `${key.toUpperCase()}=${Number(v).toFixed(4)}`;
}

async function loadTasks() {
  try {
    const resp = await fetch("/api/tasks?limit=100");
    const tasks = await resp.json();
    if (!resp.ok) throw new Error(tasks.detail || "加载失败");
    countEl.textContent = tasks.length;
    if (!tasks.length) {
      emptyHint.hidden = false;
      tbody.innerHTML = "";
      return;
    }
    emptyHint.hidden = true;
    tbody.innerHTML = tasks.map(t => {
      if (t.status === "failed") {
        return `<tr>
          <td class="mono">${esc(t.task_id)}</td>
          <td colspan="7" style="color:var(--error)">失败：${esc(t.error_message || "未知错误")}</td>
          <td><a class="btn-ghost" style="padding:4px 10px" href="/result.html?task_id=${esc(t.task_id)}">查看</a></td>
        </tr>`;
      }
      return `<tr>
        <td class="mono">${esc(t.task_id)}</td>
        <td>${esc(t.target_col || "—")}</td>
        <td>${t.task_type === "regression" ? "回归" : "分类"}</td>
        <td>${esc(t.best_model || "—")}</td>
        <td class="mono" style="color:var(--success)">${fmtScore(t.score_key, t.best_score)}</td>
        <td>${t.n_samples ?? "—"}</td>
        <td class="mono">${t.total_time_sec != null ? t.total_time_sec + "s" : "—"}</td>
        <td class="mono" style="font-size:12px">${fmtTime(t.created_at)}</td>
        <td><a class="btn-ghost" style="padding:4px 10px" href="/result.html?task_id=${esc(t.task_id)}">查看</a></td>
      </tr>`;
    }).join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="9" style="color:var(--error)">加载失败：${esc(err.message)}</td></tr>`;
  }
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

loadTasks();
