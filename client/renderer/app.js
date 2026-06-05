// Логика клиента: ассистент-анкета, стриминг протокола, плеер, маркировка, правки.

const HTTP = window.APP.backendHttp;
const WS = window.APP.backendWs;
const SESSION = "sess-" + Date.now().toString(36);

const $ = (id) => document.getElementById(id);
const state = {
  ws: null,
  recording: false,
  captures: [],
  segments: new Map(), // id -> {el, words:[{el,start,end}]}
  assistantStep: null,
};

// ---------- health + микрофоны ----------
async function checkHealth() {
  try {
    const r = await fetch(`${HTTP}/api/health`).then((x) => x.json());
    $("health").textContent = `${r.asr} · ${r.model}`;
    $("health").className = "badge ok";
  } catch {
    $("health").textContent = "backend offline";
    $("health").className = "badge err";
  }
}

async function loadMics() {
  // нужен доступ к устройствам — запросим разрешение
  try { (await navigator.mediaDevices.getUserMedia({ audio: true })).getTracks().forEach((t) => t.stop()); } catch {}
  const devices = (await navigator.mediaDevices.enumerateDevices())
    .filter((d) => d.kind === "audioinput");
  for (const sel of [$("mic0"), $("mic1")]) {
    const cur = sel.value;
    sel.innerHTML = "";
    devices.forEach((d, i) => {
      const o = document.createElement("option");
      o.value = d.deviceId;
      o.textContent = d.label || `Микрофон ${i + 1}`;
      sel.appendChild(o);
    });
    if (cur) sel.value = cur;
  }
  // по умолчанию второй микрофон для опрашиваемого, если есть
  if (devices[1]) $("mic1").value = devices[1].deviceId;
}

// ---------- AI-ассистент: анкета ----------
function speak(text) {
  try {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "ru-RU";
    window.speechSynthesis.speak(u);
  } catch {}
}

function showStep(step) {
  state.assistantStep = step;
  const promptEl = $("assistantPrompt");
  promptEl.textContent = "🤖 " + step.prompt;
  promptEl.classList.add("show");
  speak(step.prompt);
  $("startAssistant").hidden = true;
  if (step.needs_answer) {
    $("assistantControls").hidden = false;
    $("answerInput").value = "";
    $("answerInput").focus();
  } else {
    // info-шаг — кнопка «далее»
    $("assistantControls").hidden = true;
    setTimeout(advanceInfo, 1200);
  }
}

async function advanceInfo() {
  const r = await fetch(`${HTTP}/api/assistant/next`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION }),
  }).then((x) => x.json());
  handleNext(r);
}

function handleNext(r) {
  renderFields(r.fields);
  if (r.next.finished) {
    $("assistantPrompt").textContent = "✅ Анкета заполнена. Можно переходить к диалогу.";
    $("assistantControls").hidden = true;
    return;
  }
  showStep(r.next);
}

async function startAssistant() {
  const step = await fetch(`${HTTP}/api/assistant/start`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION }),
  }).then((x) => x.json());
  showStep(step);
}

async function confirmAnswer() {
  const answer = $("answerInput").value.trim();
  if (!answer) return;
  const r = await fetch(`${HTTP}/api/assistant/answer`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION, answer }),
  }).then((x) => x.json());
  handleNext(r);
}

let answerRec = null;
async function toggleRecordAnswer() {
  const btn = $("recordAnswer");
  if (!answerRec) {
    answerRec = new window.AudioCapture.OneShotRecorder($("mic1").value);
    await answerRec.start();
    btn.textContent = "⏹ Остановить";
    btn.style.background = "#c5384a"; btn.style.color = "#fff";
  } else {
    const pcm = answerRec.stop();
    answerRec = null;
    btn.textContent = "🎤 Ответ голосом";
    btn.style.background = ""; btn.style.color = "";
    const r = await fetch(`${HTTP}/api/transcribe`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" },
      body: pcm.buffer,
    }).then((x) => x.json());
    $("answerInput").value = r.text || "";
  }
}

function renderFields(fields) {
  const tb = $("fieldsTable").querySelector("tbody");
  tb.innerHTML = "";
  (fields || []).forEach((f) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="k">${f.label}</td>
      <td class="v ${f.value ? "filled" : ""}">${f.value || "—"}</td>`;
    tb.appendChild(tr);
  });
}

// ---------- Протокол: стриминг ----------
function startRecording() {
  state.ws = new WebSocket(`${WS}/ws/stream/${SESSION}`);
  state.ws.binaryType = "arraybuffer";
  state.ws.onopen = async () => {
    state.ws.send(JSON.stringify({ type: "start" }));
    const mics = [
      { ch: 0, dev: $("mic0").value },
      { ch: 1, dev: $("mic1").value },
    ];
    for (const m of mics) {
      if (!m.dev) continue;
      const cap = new window.AudioCapture.CaptureChannel(m.ch, m.dev, state.ws);
      await cap.start();
      state.captures.push(cap);
    }
    state.recording = true;
    $("startRec").disabled = true;
    $("stopRec").disabled = false;
  };
  state.ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "segment") addSegment(msg);
  };
}

function stopRecording() {
  state.captures.forEach((c) => c.stop());
  state.captures = [];
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "stop" }));
  }
  state.recording = false;
  $("startRec").disabled = false;
  $("stopRec").disabled = true;
}

function speakerClass(ch) {
  return ch === 0 ? "ch0" : ch === 1 ? "ch1" : "auto";
}

function addSegment(seg) {
  const wrap = document.createElement("div");
  wrap.className = "segment";
  wrap.dataset.id = seg.id;

  const sp = document.createElement("div");
  sp.className = "seg-speaker " + speakerClass(seg.channel);
  sp.textContent = seg.speaker + ":";
  sp.title = "Клик — изменить метку голоса";
  sp.onclick = () => openSpeakerModal(seg.channel);

  const body = document.createElement("div");
  body.className = "seg-body";

  const txt = document.createElement("div");
  txt.className = "seg-text";
  txt.contentEditable = "true";
  const words = [];
  seg.words.forEach((w) => {
    const span = document.createElement("span");
    span.className = "word";
    span.textContent = w.text;
    span.dataset.start = w.start;
    span.dataset.end = w.end;
    span.onclick = (ev) => { ev.stopPropagation(); seekTo(w.start); };
    txt.appendChild(span);
    words.push({ el: span, start: w.start, end: w.end });
  });
  if (!seg.words.length) txt.textContent = seg.text;

  txt.addEventListener("blur", () => saveEdit(seg.id, txt, wrap));

  const meta = document.createElement("div");
  meta.className = "seg-meta";
  meta.textContent = `${fmt(seg.start)}–${fmt(seg.end)}`;

  body.appendChild(txt);
  body.appendChild(meta);
  wrap.appendChild(sp);
  wrap.appendChild(body);
  $("transcript").appendChild(wrap);
  $("transcript").scrollTop = $("transcript").scrollHeight;

  state.segments.set(seg.id, { el: wrap, speakerEl: sp, channel: seg.channel, words, meta });
}

async function saveEdit(id, txtEl, wrap) {
  const text = txtEl.innerText.trim();
  const r = await fetch(`${HTTP}/api/segment/edit`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION, segment_id: id, text }),
  }).then((x) => x.json());
  if (r.ok) {
    const s = state.segments.get(id);
    if (s && !s.meta.querySelector(".edited")) {
      const tag = document.createElement("span");
      tag.className = "edited";
      tag.textContent = "  · ред.";
      s.meta.appendChild(tag);
    }
  }
}

// ---------- Маркировка спикеров ----------
let modalChannel = null;
function openSpeakerModal(channel) {
  modalChannel = channel;
  $("modalChannel").textContent = channel;
  $("speakerInput").value = "";
  $("speakerModal").hidden = false;
}
async function applySpeaker(label) {
  if (!label) label = $("speakerInput").value.trim();
  if (!label || modalChannel === null) return;
  await fetch(`${HTTP}/api/speaker`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION, channel: modalChannel, label }),
  });
  // обновляем все сегменты этого канала локально
  state.segments.forEach((s) => {
    if (s.channel === modalChannel) s.speakerEl.textContent = label + ":";
  });
  $("speakerModal").hidden = true;
}

// ---------- Плеер: привязка аудио ↔ текст ----------
function fmt(t) {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
function seekTo(t) {
  const a = $("audio");
  if (a.src) { a.currentTime = t; a.play(); }
}
async function loadAudio() {
  // дописываем WAV на сервере и загружаем
  $("audio").src = `${HTTP}/api/audio/${SESSION}?t=${Date.now()}`;
  $("audio").load();
}
function onTimeUpdate() {
  const t = $("audio").currentTime;
  state.segments.forEach((s) => {
    let active = false;
    s.words.forEach((w) => {
      const playing = t >= w.start && t < w.end;
      w.el.classList.toggle("playing", playing);
      if (playing) active = true;
    });
    s.el.classList.toggle("active", active);
  });
}

// ---------- bind ----------
function bind() {
  $("startAssistant").onclick = startAssistant;
  $("confirmAnswer").onclick = confirmAnswer;
  $("recordAnswer").onclick = toggleRecordAnswer;
  $("startRec").onclick = startRecording;
  $("stopRec").onclick = stopRecording;
  $("saveBtn").onclick = () =>
    fetch(`${HTTP}/api/save`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION }),
    }).then(() => alert("Протокол сохранён на сервере."));
  $("loadAudio").onclick = loadAudio;
  $("refreshMics").onclick = loadMics;
  $("audio").addEventListener("timeupdate", onTimeUpdate);

  $("speakerCancel").onclick = () => ($("speakerModal").hidden = true);
  $("speakerApply").onclick = () => applySpeaker();
  document.querySelectorAll("#speakerModal .presets button").forEach((b) => {
    b.onclick = () => applySpeaker(b.dataset.v);
  });
}

bind();
checkHealth();
loadMics();
