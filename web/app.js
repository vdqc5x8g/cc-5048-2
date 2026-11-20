"use strict";

// All spatial math happens server-side in the shfield kernel. This file only
// collects parameters, calls /api/compute, draws returned maps and plays the
// rendered WAV. Kernel errors are shown verbatim in the result panel.

const $ = (id) => document.getElementById(id);
const state = {
  sources: [
    { name: "front", theta: 90, phi: 0 },
    { name: "left", theta: 90, phi: 90 },
  ],
  lastResult: null,
  audioBuffer: null,
  audioCtx: null,
  activeSource: null,
};

const deg2rad = (d) => (d * Math.PI) / 180;

function renderSourceTable() {
  const tbody = $("sourceTable").querySelector("tbody");
  tbody.innerHTML = "";
  state.sources.forEach((s, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input data-k="name" value="${s.name}" /></td>
      <td><input data-k="theta" type="number" min="0" max="180" step="1" value="${s.theta}" /></td>
      <td><input data-k="phi" type="number" min="-180" max="180" step="1" value="${s.phi}" /></td>
      <td><button data-act="del" type="button">删除</button></td>`;
    tr.querySelectorAll("input").forEach((inp) => {
      inp.addEventListener("change", () => {
        const key = inp.dataset.k;
        s[key] = key === "name" ? inp.value : Number(inp.value);
      });
    });
    tr.querySelector('[data-act="del"]').addEventListener("click", () => {
      if (state.sources.length > 1) {
        state.sources.splice(i, 1);
        renderSourceTable();
      }
    });
    tbody.appendChild(tr);
  });
}

function syncOrderLabel() {
  const n = Number($("order").value);
  $("orderLabel").textContent = n;
  $("channelLabel").textContent = (n + 1) ** 2;
}
["yaw", "pitch", "roll"].forEach((id) => {
  $(id).addEventListener("input", () => $(id + "Label").textContent = $(id).value);
});

function collectPayload() {
  return {
    scene: $("scene").value,
    duration: Number($("duration").value),
    seed: Number($("seed").value),
    order: Number($("order").value),
    yaw: Number($("yaw").value),
    pitch: Number($("pitch").value),
    roll: Number($("roll").value),
    layout: $("layout").value,
    binaural: $("binaural").checked,
    sources: state.sources.map((s) => ({ ...s })),
  };
}

function setStatus(text, ok) {
  const el = $("status");
  el.textContent = text;
  el.className = "status " + (ok === undefined ? "" : ok ? "ok" : "err");
}

// ---- canvas: equirectangular field power map + source/speaker markers ----
function drawMap(result) {
  const canvas = $("map");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const map = result.power_map;
  const nT = map.length, nP = map[0].length;
  let lo = Infinity, hi = -Infinity;
  map.forEach((row) => row.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
  if (hi - lo < 1e-12) hi = lo + 1e-12;

  const cw = W / nP, ch = H / nT;
  for (let i = 0; i < nT; i++) {
    for (let j = 0; j < nP; j++) {
      const t = (map[i][j] - lo) / (hi - lo);
      // dark blue -> cyan -> yellow heat scale
      const r = Math.round(40 + 215 * Math.max(0, t - 0.5) * 2);
      const g = Math.round(60 + 180 * t);
      const b = Math.round(120 + 100 * (1 - t));
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(j * cw, i * ch, cw + 1, ch + 1);
    }
  }

  // phi in [-180,180] maps x 0..W; theta in [0,180] maps y 0..H.
  const project = (thetaDeg, phiDeg) => [
    ((phiDeg + 180) / 360) * W,
    (thetaDeg / 180) * H,
  ];

  // grid lines every 30 deg
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  for (let a = -180; a <= 180; a += 30) {
    const [x] = project(0, a);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let t = 0; t <= 180; t += 30) {
    const [, y] = project(t, 0);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  // speakers
  result.layout.speakers.forEach((spk) => {
    const [x, y] = project(spk.theta_deg, spk.phi_deg);
    ctx.fillStyle = "#ef476f";
    ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.fill();
  });

  // sources (listener frame) with name labels
  ctx.font = "12px sans-serif";
  result.sources_listener_frame.forEach((s) => {
    const [x, y] = project(s.theta_deg, s.phi_deg);
    ctx.fillStyle = "#ffd166";
    ctx.beginPath();
    ctx.moveTo(x, y - 8); ctx.lineTo(x + 7, y + 5); ctx.lineTo(x - 7, y + 5);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.fillText(s.name, x + 9, y + 4);
  });
}

function formatResults(r) {
  const rot = r.rotation;
  const lines = [
    `阶数 N=${r.order}，ACN/SN3D 声道数=${r.channels}，场景=${r.scene}`,
    `旋转 yaw/pitch/roll = ${rot.yaw_deg.toFixed(1)} / ${rot.pitch_deg.toFixed(1)} / ${rot.roll_deg.toFixed(1)}°`,
    `旋转矩阵正交误差 max|D Dᵀ-I| = ${rot.orthogonality_max_err.toExponential(2)}`,
    `各阶能量增益 = [${rot.per_order_rms_gain.map((g) => g.toFixed(6)).join(", ")}]`,
    `球面能量 旋转前=${rot.spherical_energy_before.toExponential(4)} 旋转后=${rot.spherical_energy_after.toExponential(4)}`,
    `能量相对漂移 = ${rot.relative_energy_drift.toExponential(2)}`,
    `解码布局=${r.layout.name}（实际 ${r.layout.effective}），双耳=${r.layout.binaural}`,
    `输出通道 rms: ${r.layout.feed_labels.map((l, i) => `${l}=${r.feed_rms[i].toFixed(4)}`).join("  ")}`,
    `各阶能量: ${Object.entries(r.energy_per_order).map(([l, v]) => `l${l}=${v.toExponential(3)}`).join("  ")}`,
    `声源听者相对方向: ${r.sources_listener_frame
      .map((s) => `${s.name}(theta=${s.theta_deg.toFixed(1)}°, phi=${s.phi_deg.toFixed(1)}°)`)
      .join("; ")}`,
  ];
  return lines.join("\n");
}

async function compute() {
  setStatus("内核计算中（编码 → 旋转 → 解码）…", undefined);
  $("compute").disabled = true;
  try {
    const resp = await fetch("/api/compute", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "audio/wav" },
      body: JSON.stringify(collectPayload()),
    });
    if (!resp.ok) {
      const err = await resp.json();
      setStatus(`球谐内核报错（${err.kind}）：\n${err.error}`, false);
      $("results").textContent = "";
      return;
    }
    const meta = JSON.parse(resp.headers.get("X-shfield-meta"));
    const wavBytes = await resp.arrayBuffer();
    state.lastResult = meta;
    drawMap(meta);
    $("results").textContent = formatResults(meta);
    setStatus(
      `完成：${meta.channels} 个 SH 声道 → ${meta.layout.feed_labels.length} 个重放通道。` +
        `旋转正交误差 ${meta.rotation.orthogonality_max_err.toExponential(1)}，` +
        `能量漂移 ${meta.rotation.relative_energy_drift.toExponential(1)}。可试听。`,
      true
    );
    state.audioBuffer = await decodeWav(wavBytes);
    $("play").disabled = false;
  } catch (err) {
    setStatus("请求失败：" + err.message, false);
  } finally {
    $("compute").disabled = false;
  }
}

async function decodeWav(arrayBuffer) {
  if (!state.audioCtx) state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return await state.audioCtx.decodeAudioData(arrayBuffer.slice(0));
}

function play() {
  if (!state.audioBuffer) return;
  stopPlayback();
  const ctx = state.audioCtx;
  const node = ctx.createBufferSource();
  node.buffer = state.audioBuffer;
  node.connect(ctx.destination);
  node.start();
  state.activeSource = node;
  node.onended = () => { state.activeSource = null; $("stop").disabled = true; };
  $("stop").disabled = false;
}

function stopPlayback() {
  if (state.activeSource) {
    try { state.activeSource.stop(); } catch (_) {}
    state.activeSource = null;
  }
  $("stop").disabled = true;
}

async function loadOptions() {
  const opts = await (await fetch("/api/options")).json();
  const sel = $("layout");
  opts.layouts.forEach((name) => {
    const o = document.createElement("option");
    o.value = name; o.textContent = name;
    sel.appendChild(o);
  });
  sel.value = "octagon";
}

function wireEvents() {
  $("order").addEventListener("input", syncOrderLabel);
  $("addSource").addEventListener("click", () => {
    state.sources.push({ name: `src${state.sources.length + 1}`, theta: 90, phi: 0 });
    renderSourceTable();
  });
  $("presetFront").addEventListener("click", () => {
    state.sources = [{ name: "front", theta: 90, phi: 0 }];
    renderSourceTable();
  });
  $("presetFour").addEventListener("click", () => {
    state.sources = [
      { name: "front", theta: 90, phi: 0 },
      { name: "right", theta: 90, phi: -90 },
      { name: "back", theta: 90, phi: 180 },
      { name: "left", theta: 90, phi: 90 },
    ];
    renderSourceTable();
  });
  $("compute").addEventListener("click", compute);
  $("play").addEventListener("click", play);
  $("stop").addEventListener("click", stopPlayback);
}

loadOptions();
renderSourceTable();
syncOrderLabel();
wireEvents();

// ---------------------------------------------------------------------------
// Diagnostic view: stage substitution. The server (shfield kernel) computes
// everything; this file only lays it out and draws the per-order bar chart.
// ---------------------------------------------------------------------------

function fmtRel(v) {
  if (v === null || v === undefined) return "—";
  if (v < 1e-12) return "0";
  return v.toExponential(1);
}

function firstLabel(v) {
  return v === null || v === undefined ? "—" : `l=${v}`;
}

function runDiagnostic() {
  const payload = {
    order: Number($("diagOrder").value),
    yaw: Number($("diagYaw").value),
    pitch: Number($("diagPitch").value),
    bypass_import: $("bypassImport").checked,
    bypass_normalization: $("bypassNorm").checked,
    bypass_rotation: $("bypassRot").checked,
  };
  $("diagStatus").textContent = "内核计算中（逐段替换）…";
  $("diagStatus").className = "status";
  $("runDiag").disabled = true;
  fetch("/api/diagnose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then(async (resp) => {
      const body = await resp.json();
      if (!resp.ok || !body.ok) {
        throw new Error(body.error || `HTTP ${resp.status}`);
      }
      return body;
    })
    .then((r) => {
      drawStageTable(r);
      drawMotionTable(r);
      drawOrderChart(r);
      $("diagStatus").textContent =
        `完成 N=${r.order}：旋转独立参照 ${r.rotation.pass === null ? "已旁路" : r.rotation.pass ? "通过" : "失败"}；` +
        `零阶负对照 ${r.omni_control.pass ? "通过" : "失败"}；${r.confusion.attribution}`;
      $("diagStatus").className =
        "status " + (r.omni_control.pass && r.rotation.pass !== false ? "ok" : "err");
    })
    .catch((err) => {
      $("diagStatus").textContent = "诊断失败：" + err.message;
      $("diagStatus").className = "status err";
    })
    .finally(() => { $("runDiag").disabled = false; });
}

function drawStageTable(r) {
  const tbody = $("stageTable").querySelector("tbody");
  tbody.innerHTML = "";
  r.stages.forEach((s) => {
    const tr = document.createElement("tr");
    const first = s.first_deviating_order;
    const firstCls = first === null || first === undefined ? "pass" : "fail first";
    let cells = `<td title="${s.variant}">${s.label}</td>`;
    if (!s.applied) {
      cells += `<td colspan="5" class="muted">已旁路</td>`;
    } else {
      cells += `<td class="${firstCls}">${firstLabel(first)}</td>`;
      for (const l of ["0", "1", "2", "3"]) {
        const v = s.per_order_relative[l];
        const cls = v === undefined ? "" : v > r.tolerance ? "fail" : "pass";
        cells += `<td class="${cls}">${fmtRel(v)}</td>`;
      }
    }
    tr.innerHTML = cells;
    tbody.appendChild(tr);
  });
}

function drawMotionTable(r) {
  const tbody = $("motionTable").querySelector("tbody");
  tbody.innerHTML = "";
  r.motion.poses.forEach((p) => {
    const tr = document.createElement("tr");
    if (p.reverse_restore === null) {
      tr.innerHTML = `<td>${p.name}</td><td colspan="2" class="muted">旋转已旁路</td>`;
    } else {
      const rev = p.reverse_restore.pass;
      const same = p.same_order_restore.pass;
      tr.innerHTML =
        `<td>${p.name}</td>` +
        `<td class="${rev ? "pass" : "fail"}">${rev ? "回起点" : "偏离"}</td>` +
        `<td class="${same ? "pass" : "fail"}">${same ? "回起点" : "偏离"}</td>`;
    }
    tbody.appendChild(tr);
  });
  const red = r.motion.same_order_pose_reduction;
  $("poseReduction").textContent = red.reduced_still_fails
    ? `缩减后仍会失败的最小姿态集（同序还原）：${red.reduced.join("、")}`
    : "同序还原在全部姿态均通过。";

  const omni = r.omni_control;
  $("diagSummary").textContent = [
    `零阶（全向）负对照：能量 旋转前=${omni.energy_before.toExponential(4)} ` +
      `旋转后=${omni.energy_after.toExponential(4)}，相对漂移=${omni.relative_drift.toExponential(2)}，` +
      `通过=${omni.pass}`,
    `首个偏离阶次（按环节）：` +
      Object.entries(r.confusion.first_deviating_by_variant)
        .map(([k, v]) => `${k}=${firstLabel(v)}`)
        .join("  "),
    `归因：${r.confusion.attribution}`,
  ].join("\n");
}

function drawOrderChart(r) {
  const canvas = $("orderChart");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const stages = r.stages.filter((s) => s.applied && s.per_order_relative);
  const orders = Object.keys(stages[0] ? stages[0].per_order_relative : {});
  if (!stages.length) {
    ctx.fillStyle = "#93a1b0";
    ctx.font = "13px sans-serif";
    ctx.fillText("所有输入环节已旁路。", 12, 24);
    return;
  }
  const padL = 120, padB = 26, padT = 14;
  const plotW = W - padL - 16, plotH = H - padB - padT;
  let hi = r.tolerance * 3;
  stages.forEach((s) => Object.values(s.per_order_relative).forEach((v) => { hi = Math.max(hi, v); }));
  const yOf = (v) => padT + plotH * (1 - Math.min(v, hi) / hi);

  // tolerance line
  const yTol = yOf(r.tolerance);
  ctx.strokeStyle = "#ff6b6b";
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(padL, yTol); ctx.lineTo(W - 16, yTol); ctx.stroke();
  ctx.setLineDash([]);

  const groupW = plotW / orders.length;
  const barW = Math.max(3, (groupW - 12) / stages.length);
  const colors = ["#4ea1ff", "#ffd166", "#57d18a", "#ef476f", "#b388ff"];
  ctx.font = "11px sans-serif";
  orders.forEach((l, gi) => {
    stages.forEach((s, si) => {
      const v = s.per_order_relative[l] || 0;
      const x = padL + gi * groupW + 6 + si * barW;
      const y = yOf(v);
      ctx.fillStyle = colors[si % colors.length];
      ctx.fillRect(x, y, barW - 1, padT + plotH - y);
    });
    ctx.fillStyle = "#93a1b0";
    ctx.fillText(`l${l}`, padL + gi * groupW + groupW / 2 - 6, H - 8);
  });
  stages.forEach((s, si) => {
    ctx.fillStyle = colors[si % colors.length];
    ctx.fillRect(8, padT + si * 16, 10, 10);
    ctx.fillStyle = "#e6edf3";
    ctx.fillText(s.variant, 22, padT + 9 + si * 16);
  });
}

$("runDiag").addEventListener("click", runDiagnostic);
