/* Frontend logic: submit job, poll status every ~1s, render stages/logs/result. */
"use strict";

const STAGES = [
  ["download", "Video Download"],
  ["audio_extraction", "Audio Extraction"],
  ["coarse_asr", "Coarse ASR"],
  ["candidate_generation", "Candidate Generation"],
  ["fine_validation", "Fine ASR Validation"],
  ["alignment", "Forced Alignment"],
  ["frame_extraction", "Frame Extraction"],
];

const $ = (id) => document.getElementById(id);

let jobId = null;
let pollTimer = null;
const startedAt = Date.now();

function fmtSeconds(s) {
  if (s == null || isNaN(s)) return "\u2014";
  return s >= 100 ? Math.round(s) + "s" : Number(s).toFixed(1) + "s";
}

function badge(state, labelOverride) {
  const el = document.createElement("span");
  el.className = `badge badge-${state}`;
  el.textContent = labelOverride || state;
  return el;
}

function renderStages(stageStates, currentState, message) {
  const list = $("stage-list");
  list.innerHTML = "";
  for (const [key, label] of STAGES) {
    const li = document.createElement("li");
    li.className = "stage-row";

    const name = document.createElement("span");
    name.className = "stage-name";
    name.textContent = label;

    const right = document.createElement("span");
    right.className = "stage-msg";
    const info = stageStates[key] || { state: "pending" };
    if (key === currentState) {
      right.textContent = message || info.message || "";
      right.title = right.textContent;
    }
    let badgeState = info.state === "done" ? "done" : info.state;
    if (badgeState === "pending" && key === currentState) badgeState = "running";
    name.appendChild(document.createTextNode(" "));
    li.appendChild(name);
    li.appendChild(right);
    li.appendChild(badge(badgeState));
    list.appendChild(li);
  }
  // Final pseudo-stage
  const li = document.createElement("li");
  li.className = "stage-row";
  const doneLabel = document.createElement("span");
  doneLabel.className = "stage-name";
  doneLabel.textContent = "Complete";
  li.appendChild(doneLabel);
  li.appendChild(badge("done", "Complete"));
  if (currentState !== null && document.getElementById("result-card") &&
      !$("error-banner").hidden === false) {
    /* noop placeholder to keep DOM simple */
  }
  list.appendChild(li);
  // Hide the Complete row while the job is still running:
  if (!jobDone) list.lastChild.style.display = "none";
}

function renderLog(lines) {
  const panel = $("log-panel");
  panel.textContent = lines.join("\n");
  if ($("autoscroll").checked) panel.scrollTop = panel.scrollHeight;
}

let jobDone = false;

function renderResult(data) {
  jobDone = true;
  $("run-btn").disabled = false;
  $("busy-hint").hidden = true;
  renderStages(data.stages, data.current_stage, data.message);
  renderLog(data.logs);

  if (data.state === "failed") {
    $("current-stage").textContent = "Failed";
    const banner = $("error-banner");
    banner.textContent = data.error || "Unknown error";
    banner.hidden = false;
    $("result-card").hidden = false;
    $("frame-img").hidden = true;
    return;
  }
  if (data.state !== "done" || !data.result) return;

  $("current-stage").textContent = "Complete";
  const r = data.result;

  const banner = $("error-banner");
  banner.hidden = true;

  $("matched-text").textContent = r.matched_text || "";
  $("timestamp-hms").textContent = r.timestamp_hhmmss || "";
  $("frame-number").textContent = r.frame_number ?? "\u2014";
  $("similarity").textContent =
    r.similarity != null ? Number(r.similarity).toFixed(4) : "\u2014";
  $("total-runtime").textContent = fmtSeconds(data.elapsed_s);
  $("timestamp-s").textContent =
    r.timestamp != null ? Number(r.timestamp).toFixed(3) + "s" : "\u2014";

  // Badges
  const badges = $("badges");
  badges.innerHTML = "";
  badges.appendChild(badge("done", `tier: ${r.fallback_tier || "?"}`));
  badges.appendChild(badge("done", r.aligned ? "aligned: yes" : "aligned: no"));

  // Secondary stats from cascade timings (keys: coarse_asr_<model>_s,
  // fine_asr_validation_t0_s, forced_alignment_s)
  const t = r.stage_timings || {};
  const coarseKey = Object.keys(t).find((k) => k.startsWith("coarse_asr_"));
  const fineKey = Object.keys(t).find((k) => k.startsWith("fine_asr_validation"));
  $("t-coarse").textContent = coarseKey ? fmtSeconds(t[coarseKey]) : "\u2014";
  $("t-fine").textContent = fineKey ? fmtSeconds(t[fineKey]) : "\u2014";
  $("t-align").textContent = t.forced_alignment_s != null
    ? fmtSeconds(t.forced_alignment_s) : "\u2014";
  $("tier").textContent = r.fallback_tier || "\u2014";
  $("probs").textContent =
    `${Number(r.average_word_probability ?? 0).toFixed(3)} / ` +
    `${Number(r.minimum_word_probability ?? 0).toFixed(3)}`;

  // Frame image via /frames static mount.
  if (r.frame_image_url) {
    $("frame-img").src = r.frame_image_url;
    $("frame-img").hidden = false;
    $("frame-caption").textContent = r.frame_image_path || "";
  }

  $("result-card").hidden = false;
}

function poll() {
  if (!jobId) return;
  fetch(`/api/jobs/${jobId}`)
    .then((resp) => {
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return resp.json();
    })
    .then((data) => {
      jobDone = data.state === "done" || data.state === "failed";
      $("elapsed").textContent = `Elapsed: ${fmtSeconds(data.elapsed_s)}s`
        .replace(/\s*s$/, "s");
      $("current-stage").textContent =
        data.state === "running"
          ? `${data.current_stage || ""} \u2014 ${data.message || ""}`
          : data.state.charAt(0).toUpperCase() + data.state.slice(1);
      renderStages(data.stages || {}, data.current_stage, data.message);
      renderLog(data.logs || []);
      if (jobDone) {
        clearInterval(pollTimer);
        renderResult(data);
      }
    })
    .catch(() => { /* transient network hiccup: keep polling */ });
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(poll, 1000);
  poll();
}

document.getElementById("run-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const url = $("url").value.trim();
  const target = $("target").value.trim();
  if (!url || !target) {
    alert("Please provide both a video URL and target dialogue.");
    return;
  }

  $("run-btn").disabled = true;
  $("status-card").hidden = false;
  $("result-card").hidden = true;
  $("error-banner").hidden = true;
  jobDone = false;

  fetch("/api/localize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, target }),
  })
    .then(async (resp) => {
      const body = await resp.json();
      if (!resp.ok) {
        throw new Error(body.detail || `HTTP ${resp.status}`);
      }
      jobId = body.job_id;
      startPolling();
    })
    .catch((err) => {
      $("run-btn").disabled = false;
      $("status-card").hidden = true;
      const banner = $("error-banner");
      banner.textContent = err.message === "HTTP 409"
        ? "A pipeline job is already running. Please wait."
        : `Error: ${err.message}`;
      banner.hidden = false;
      $("result-card").hidden = false;
    });
});
