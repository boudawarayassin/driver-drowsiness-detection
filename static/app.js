"use strict";

// ── DOM refs ──────────────────────────────────────
const el = (id) => document.getElementById(id);

const videoImg = el("video-img");
const idleScreen = el("idle-screen");
const connBadge = el("conn-badge");
const connLabel = el("conn-label");
const statusOrb = el("status-orb");
const statusText = el("status-text");
const eyeBar = el("eye-bar");
const yawnBar = el("yawn-bar");
const eyeVal = el("eye-val");
const yawnVal = el("yawn-val");
const logBody = el("log-body");
const btnStart = el("btn-start");
const btnStop = el("btn-stop");
const chipFps = el("chip-fps");
const chipTime = el("chip-time");
const stAlerts = el("st-alerts");
const stFps = el("st-fps");
const stTime = el("st-time");

// ── State ─────────────────────────────────────────
let ws = null;
let running = false;

// ── Session control ───────────────────────────────
function startSession() {
  if (running) return;
  running = true;

  btnStart.disabled = true;
  btnStop.disabled = false;

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    connBadge.classList.add("online");
    connLabel.textContent = "ONLINE";
    videoImg.classList.remove("hidden");
    idleScreen.classList.add("hidden");
  };

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    const s = data.state;

    // Update image
    videoImg.src = "data:image/jpeg;base64," + data.frame;

    // Update UI elements
    statusText.textContent = s.status;
    chipFps.textContent = `${s.fps} FPS`;
    stFps.textContent = s.fps;
    chipTime.textContent = s.session_time;
    stTime.textContent = s.session_time;
    stAlerts.textContent = s.total_alerts;

    // Meters
    const eyePct = (s.closed_frames / s.threshold) * 100;
    eyeBar.style.width = `${Math.min(eyePct, 100)}%`;
    eyeVal.textContent = s.closed_frames;

    const yawnPct = (s.yawn_frames / 10) * 100;
    yawnBar.style.width = `${Math.min(yawnPct, 100)}%`;
    yawnVal.textContent = s.yawn_frames;

    // Log update
    if (s.log && s.log.length > 0) {
      logBody.innerHTML = s.log
        .map(
          (item) => `
        <div class="log-item ${item.level}">
          <span class="log-t">${item.time}</span>
          <span>${item.msg}</span>
        </div>
      `,
        )
        .join("");
    }
  };

  ws.onclose = () => {
    stopSession();
  };
}

function stopSession() {
  running = false;
  if (ws) ws.close();
  btnStart.disabled = false;
  btnStop.disabled = true;
  connBadge.classList.remove("online");
  connLabel.textContent = "OFFLINE";
  videoImg.classList.add("hidden");
  idleScreen.classList.remove("hidden");
  location.reload();
}

// Expose functions to HTML
window.startSession = startSession;
window.stopSession = stopSession;
