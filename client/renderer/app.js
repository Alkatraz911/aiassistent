// Логика клиента: ассистент-анкета, стриминг протокола, плеер, маркировка, правки.

const HTTP = window.APP.backendHttp;
const WS = window.APP.backendWs;
const SESSION = "sess-" + Date.now().toString(36);

const $ = (id) => document.getElementById(id);
const state = {
  ws: null,
  recording: false,
  captures: [],
  segments: new Map(),      // id -> {el, words:[{el,start,end}]}
  liveSegments: new Map(),  // channel -> {el, txtEl, channel, utteranceId} — ещё не финализированные реплики
  assistantStep: null,
  flushed: true,            // false между отправкой "stop" и получением "stopped"
  templateId: null,         // выбранный шаблон анкеты (Блок 5)
  editingTemplateId: null,  // null = создаём новый, иначе редактируем существующий
};

// ---------- health + микрофоны ----------
async function checkHealth() {
  try {
    const r = await fetch(`${HTTP}/api/health`).then((x) => x.json());
    // Пока идёт преполёт, устройство в ответе — ещё только НАМЕРЕНИЕ: если GPU не поднимется,
    // сервер откатится на CPU. Зелёный бейдж «large-v3 · GPU» в этот момент — прямая
    // дезинформация, поэтому ждём итога и перезапрашиваем.
    if (r.loading) {
      $("health").textContent = "⏳ загрузка модели…";
      $("health").className = "badge warn";
      $("health").title = r.warning || "";
      setTimeout(checkHealth, 1500);
      return;
    }
    // Устройство показываем прямо в бейдже: одна и та же модель на CPU и на GPU — это разные
    // режимы работы (large-v3 на CPU live не тянет вовсе), и видеть это надо ДО записи, а не
    // потом по растущей задержке. r.warning непустой — GPU просили, но он не поднялся.
    const dev = (r.device || "cpu").startsWith("cuda") ? "GPU" : "CPU";
    $("health").textContent = `${r.asr} · ${r.model} · ${dev}`;
    $("health").className = r.warning ? "badge warn" : "badge ok";
    $("health").title = r.warning || "";
  } catch {
    $("health").textContent = "backend offline";
    $("health").className = "badge err";
  }
}

// ---------- выбор модели ASR (Блок 4) ----------
async function loadModels() {
  try {
    const r = await fetch(`${HTTP}/api/models`).then((x) => x.json());
    const sel = $("modelSelect");
    sel.innerHTML = "";
    (r.models || []).forEach((m) => {
      const o = document.createElement("option");
      o.value = m; o.textContent = m + (r.loaded && r.loaded.includes(m) ? " ✓" : "");
      sel.appendChild(o);
    });
    sel.value = r.active;
  } catch {}
}

async function onModelChange() {
  const sel = $("modelSelect");
  const prev = sel.dataset.active || sel.value;
  const model = sel.value;
  sel.disabled = true;
  $("health").textContent = "⏳ Загрузка модели…";
  try {
    const r = await fetch(`${HTTP}/api/models/switch`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    });
    const data = await r.json();
    if (!r.ok) {
      alert(data.error || "Не удалось переключить модель");
      sel.value = prev;
    } else {
      sel.dataset.active = data.active;
      await checkHealth();
    }
  } catch (e) {
    alert("Ошибка переключения модели: " + e);
    sel.value = prev;
  } finally {
    updateModelSelectEnabled();
  }
}

function updateModelSelectEnabled() {
  // Переключение модели запрещено во время активной записи — источник истины сервер (409),
  // но дублируем блокировку на клиенте для мгновенной обратной связи.
  $("modelSelect").disabled = state.recording;
}

async function loadMics() {
  // нужен доступ к устройствам — запросим разрешение
  try { (await navigator.mediaDevices.getUserMedia({ audio: true })).getTracks().forEach((t) => t.stop()); } catch {}
  const all = (await navigator.mediaDevices.enumerateDevices())
    .filter((d) => d.kind === "audioinput");

  // Windows отдаёт первыми ДВА псевдоустройства — «Default - …» и «Communications - …», и оба
  // указывают на один и тот же физический микрофон. Прежний код брал devices[0] и devices[1],
  // то есть сажал оба канала на одну железку: в протоколе выходили две одинаковые реплики с
  // одинаковыми тайм-кодами, а канальная диаризация («канал = спикер») теряла смысл целиком.
  // Псевдоустройства убираем из списка — выбирать нужно ФИЗИЧЕСКИЕ входы.
  let devices = all.filter((d) => d.deviceId !== "default" && d.deviceId !== "communications");
  // Подстраховка: если система отдаёт ТОЛЬКО псевдоустройства (встречается, пока не выдано
  // разрешение на микрофон), лучше показать что есть, чем пустой список.
  if (!devices.length) devices = all;

  // Два одинаковых USB-микрофона отдают одинаковую подпись — без номера их не различить.
  const seen = new Map();
  const nameOf = (d, i) => {
    const base = (d.label || `Микрофон ${i + 1}`).replace(/^(Default|Communications) - /, "");
    const n = (seen.get(base) || 0) + 1;
    seen.set(base, n);
    return n > 1 ? `${base} #${n}` : base;
  };
  const names = devices.map(nameOf);

  for (const sel of [$("mic0"), $("mic1")]) {
    const cur = sel.value;
    sel.innerHTML = "";
    // Канал 1 можно вообще не использовать — это режим одного общего микрофона: писать один
    // вход в оба канала бессмысленно (протокол задваивается), а разводить голоса потом будет
    // офлайн-диаризация. Без такого пункта единственное устройство неизбежно попадало в оба
    // селектора, и запись выходила задвоенной.
    if (sel.id === "mic1") {
      const none = document.createElement("option");
      none.value = "";
      none.textContent = "— не используется —";
      sel.appendChild(none);
    }
    devices.forEach((d, i) => {
      const o = document.createElement("option");
      o.value = d.deviceId;
      o.textContent = names[i];
      sel.appendChild(o);
    });
    if (cur !== null && [...sel.options].some((o) => o.value === cur)) sel.value = cur;
  }
  // Разные физические входы по умолчанию: первый — интервьюеру, второй — опрашиваемому.
  // Только если выбирать ещё не приходилось: кнопка ⟳ не должна сбрасывать ручной выбор.
  const has = (id) => devices.some((d) => d.deviceId === id);
  if (!has($("mic0").value) && devices[0]) $("mic0").value = devices[0].deviceId;
  // Второй канал ставим, только если есть ВТОРОЕ устройство. Единственный вход оставляем
  // неназначенным: пусть это будет осознанный выбор режима, а не молча задвоенная запись.
  // Ориентироваться на пустое значение тут нельзя: у mic1 пусто — это ещё и законный выбор
  // «— не используется —», и по нему ⟳ возвращал второе устройство, снова задваивая протокол.
  // Различает эти два случая только пометка dataset.chosen (см. onMicChosen).
  if (!$("mic1").dataset.chosen && !$("mic1").value && devices[1]) {
    $("mic1").value = devices[1].deviceId;
  }
  checkMicsDistinct();
}

// Пользователь тронул селектор микрофона — с этого момента выбор его, а не наш.
function onMicChosen(e) {
  e.currentTarget.dataset.chosen = "1";
  checkMicsDistinct();
}

// Устройство для разовой записи ответа анкеты (говорит опрашиваемый). В обычном режиме это
// его собственный микрофон; в стерео-режиме и в режиме одного общего микрофона второго
// устройства нет вовсе — берём тот единственный, что выбран для канала 0.
function answerDeviceId() {
  if ($("stereoSplit").checked) return $("mic0").value;
  return $("mic1").value || $("mic0").value;
}

// Канальная диаризация держится ровно на одном условии: каналы пришли с РАЗНЫХ микрофонов.
// Если на обоих один вход, никакой алгоритм этого потом не разведёт — предупреждаем сразу,
// а не после записи, когда протокол уже задвоился.
function checkMicsDistinct() {
  const stereo = $("stereoSplit").checked;
  // В стерео-режиме одно устройство — это норма, а не ошибка: участники разведены по L/R,
  // а не по разным входам. Второй селектор в этом режиме не участвует.
  $("mic1").disabled = stereo;
  const warn = $("micWarn");
  if (!stereo && $("mic0").value && $("mic0").value === $("mic1").value) {
    warn.textContent = "⚠ Оба канала на одном микрофоне — разбивки по голосам не будет";
    warn.hidden = false;
    return false;
  }
  if (!stereo && !$("mic1").value) {
    // Не ошибка, а осознанный режим — но напоминаем, чем разводить голоса потом.
    warn.textContent = "ℹ Один микрофон: после записи нажмите «Разметить голоса» — "
                     + "реплики разведёт диаризация";
    warn.hidden = false;
    return true;
  }
  warn.hidden = true;
  return true;
}

// ---------- AI-ассистент: анкета (Блок 6 — автозапись ответа по паузам) ----------
// speak(text, onDone) — основной путь: SpeechSynthesisUtterance.onend. Запасной таймер не
// завершает озвучку напрямую, а проверяет speechSynthesis.speaking — так безопаснее, чем
// считать TTS законченным просто по истечении времени (реальная гонка: таймер сработал бы
// раньше, чем движок TTS реально замолчал, и микрофон услышал бы хвост синтеза).
function speak(text, onDone) {
  if (!text) { if (onDone) onDone(); return; }
  try {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "ru-RU";
    let done = false;
    const finish = () => { if (done) return; done = true; if (onDone) onDone(); };
    u.onend = finish;
    u.onerror = finish;
    window.speechSynthesis.speak(u);
    const fallbackMs = Math.max(3000, text.length * 90);
    const checkFallback = () => {
      if (done) return;
      if (window.speechSynthesis.speaking) setTimeout(checkFallback, 300);
      else finish();
    };
    setTimeout(checkFallback, fallbackMs);
  } catch {
    if (onDone) onDone();
  }
}

function showStep(step) {
  state.assistantStep = step;
  const promptEl = $("assistantPrompt");
  promptEl.textContent = "🤖 " + step.prompt;
  promptEl.classList.add("show");
  $("startAssistant").hidden = true;

  if (step.needs_answer) {
    $("assistantControls").hidden = false;
    $("answerInput").value = "";
    hideAutoBoxes();
  } else {
    $("assistantControls").hidden = true;
  }

  // Сначала разъяснение (statement), затем сам вопрос (question) — озвучиваются по очереди;
  // автослушание стартует только после того, как TTS РЕАЛЬНО закончил, не по таймеру вслепую.
  const parts = [step.statement, step.question].filter(Boolean);
  const sayNext = () => {
    const text = parts.shift();
    if (text === undefined) {
      if (step.needs_answer) {
        setTimeout(startAutoListening, 250);   // защитная пауза — микрофон не услышит хвост TTS
      } else {
        advanceInfo();
      }
      return;
    }
    speak(text, sayNext);
  };
  sayNext();
}

// ---------- автозапись ответа (Блок 6) ----------
let autoRec = null;
let autoConfirmTimer = null;

function hideAutoBoxes() {
  $("autoListenBox").hidden = true;
  $("autoConfirmBox").hidden = true;
  clearInterval(autoConfirmTimer);
}

function stopAutoListening() {
  if (autoRec) { try { autoRec.stop(); } catch {} autoRec = null; }
}

function startAutoListening() {
  if (!state.assistantStep || !state.assistantStep.needs_answer) return;
  stopAutoListening();
  hideAutoBoxes();
  $("autoListenBox").hidden = false;
  autoRec = new window.AudioCapture.AutoRecorder(answerDeviceId(), {});
  autoRec.onSilence((hadSpeech) => {
    $("autoListenBox").hidden = true;
    if (hadSpeech) finishAutoListening();
    // тишина (никто не ответил за maxWaitForSpeechMs) — тихо остаёмся на ручной форме
  });
  autoRec.start().catch(() => { $("autoListenBox").hidden = true; });
}

async function finishAutoListening() {
  if (!autoRec) return;
  const pcm = autoRec.stop();
  autoRec = null;
  if (!pcm.length) return;
  const r = await fetch(`${HTTP}/api/transcribe`, {
    method: "POST", headers: { "Content-Type": "application/octet-stream" },
    body: pcm.buffer,
  }).then((x) => x.json());
  const text = (r.text || "").trim();
  if (!text) return;   // не распознано — тихо остаёмся на ручной форме, без пустого автоподтверждения
  $("answerInput").value = text;
  showAutoConfirm(text);
}

// Окно подстраховки: распознанный текст показывается несколько секунд перед автоподтверждением
// — оператор может исправить (переключиться на ручной ввод) или переслушать заново.
function showAutoConfirm(text) {
  $("autoConfirmBox").hidden = false;
  $("autoConfirmText").textContent = text;
  let secondsLeft = 2;
  const render = () => { $("autoConfirmCountdown").textContent = secondsLeft + " с"; };
  render();
  autoConfirmTimer = setInterval(() => {
    secondsLeft -= 1;
    if (secondsLeft <= 0) {
      clearInterval(autoConfirmTimer);
      $("autoConfirmBox").hidden = true;
      confirmAnswer();
    } else {
      render();
    }
  }, 1000);
  $("autoFixBtn").onclick = () => {
    clearInterval(autoConfirmTimer);
    $("autoConfirmBox").hidden = true;
    $("answerInput").focus();
  };
  $("autoRetryBtn").onclick = () => {
    clearInterval(autoConfirmTimer);
    $("autoConfirmBox").hidden = true;
    startAutoListening();
  };
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
    hideAutoBoxes();
    stopAutoListening();
    $("assistantPrompt").textContent = "✅ Анкета заполнена. Можно переходить к диалогу.";
    $("assistantControls").hidden = true;
    return;
  }
  showStep(r.next);
}

async function startAssistant() {
  const step = await fetch(`${HTTP}/api/assistant/start`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION, template_id: state.templateId }),
  }).then((x) => x.json());
  showStep(step);
}

// ---------- шаблоны анкеты (Блок 5) ----------
async function loadTemplates() {
  try {
    const list = await fetch(`${HTTP}/api/templates`).then((x) => x.json());
    const sel = $("templateSelect");
    const prev = sel.value;
    sel.innerHTML = "";
    list.forEach((t) => {
      const o = document.createElement("option");
      o.value = t.id;
      o.textContent = `${t.name}${t.is_builtin ? " (стандартный)" : ""} — ${t.step_count} шаг.`;
      sel.appendChild(o);
    });
    sel.value = list.some((t) => t.id === prev) ? prev : (list[0] ? list[0].id : "");
    state.templateId = sel.value || null;
  } catch {}
}

function stepRowTemplate(step) {
  const row = document.createElement("div");
  row.className = "step-row";
  row.innerHTML = `
    <div class="step-head">
      <input class="step-label" type="text" placeholder="Название шага" value="${escAttr(step.label || "")}" />
      <div class="step-order">
        <button type="button" class="step-up" title="Выше">▲</button>
        <button type="button" class="step-down" title="Ниже">▼</button>
      </div>
      <button type="button" class="step-remove" title="Удалить шаг">✕</button>
    </div>
    <div class="step-meta">
      <label>Тип
        <select class="step-kind">
          <option value="field">вопрос с ответом</option>
          <option value="confirm">да/нет</option>
          <option value="info">только текст</option>
        </select>
      </label>
      <label>Извлечение
        <select class="step-extractor">
          <option value="plain">как есть</option>
          <option value="fio">ФИО</option>
          <option value="birth">дата</option>
          <option value="yesno">да/нет</option>
          <option value="none">не сохранять</option>
        </select>
      </label>
    </div>
    <textarea class="step-statement" placeholder="Текст для зачитывания (разъяснение, опционально)">${step.statement || ""}</textarea>
    <textarea class="step-question" placeholder="Вопрос (если нужен ответ)">${step.question || ""}</textarea>
  `;
  row.querySelector(".step-kind").value = step.kind || "field";
  row.querySelector(".step-extractor").value = step.extractor || "plain";
  row.dataset.key = step.key || ("step" + Math.random().toString(36).slice(2, 8));
  row.querySelector(".step-remove").onclick = () => row.remove();
  row.querySelector(".step-up").onclick = () => {
    const prev = row.previousElementSibling;
    if (prev) row.parentNode.insertBefore(row, prev);
  };
  row.querySelector(".step-down").onclick = () => {
    const next = row.nextElementSibling;
    if (next) row.parentNode.insertBefore(next, row);
  };
  return row;
}

function escAttr(s) {
  return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

async function openTemplateEditor(templateId) {
  state.editingTemplateId = templateId;
  const stepsBox = $("templateSteps");
  stepsBox.innerHTML = "";
  $("templateHint").textContent = "";
  let tmpl = { name: "", description: "", steps: [], is_builtin: false };
  if (templateId) {
    tmpl = await fetch(`${HTTP}/api/templates/${templateId}`).then((x) => x.json());
  }
  $("templateModalTitle").textContent = templateId
    ? (tmpl.is_builtin ? "Копия шаблона «" + tmpl.name + "»" : "Редактировать шаблон")
    : "Новый шаблон";
  // Стандартный шаблон нельзя менять напрямую — открываем его как заготовку для копии.
  if (tmpl.is_builtin) {
    state.editingTemplateId = null;
    tmpl = { ...tmpl, name: tmpl.name + " — копия" };
    $("templateHint").textContent = "Стандартный шаблон нельзя изменить напрямую — сохранение создаст новую копию.";
  }
  $("templateName").value = tmpl.name || "";
  $("templateDescription").value = tmpl.description || "";
  (tmpl.steps || []).forEach((s) => stepsBox.appendChild(stepRowTemplate(s)));
  $("templateDeleteBtn").hidden = !state.editingTemplateId;
  $("templateModal").hidden = false;
}

function slugify(label, fallback) {
  const s = label.trim().toLowerCase()
    .replace(/[^a-zа-яё0-9]+/gi, "_").replace(/^_+|_+$/g, "");
  return s || fallback;
}

function collectStepsFromEditor() {
  const rows = [...$("templateSteps").querySelectorAll(".step-row")];
  const used = new Set();
  return rows.map((row, i) => {
    const label = row.querySelector(".step-label").value.trim() || `Шаг ${i + 1}`;
    let key = slugify(label, row.dataset.key || `step${i}`);
    while (used.has(key)) key += "_2";
    used.add(key);
    return {
      key, label,
      kind: row.querySelector(".step-kind").value,
      extractor: row.querySelector(".step-extractor").value,
      statement: row.querySelector(".step-statement").value.trim(),
      question: row.querySelector(".step-question").value.trim(),
    };
  });
}

async function saveTemplate() {
  const name = $("templateName").value.trim();
  if (!name) { $("templateHint").textContent = "Укажите название шаблона."; return; }
  const steps = collectStepsFromEditor();
  if (!steps.length) { $("templateHint").textContent = "Добавьте хотя бы один шаг."; return; }
  const body = JSON.stringify({ name, description: $("templateDescription").value.trim(), steps });
  const url = state.editingTemplateId
    ? `${HTTP}/api/templates/${state.editingTemplateId}` : `${HTTP}/api/templates`;
  const method = state.editingTemplateId ? "PUT" : "POST";
  const r = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body });
  const data = await r.json();
  if (!r.ok) { $("templateHint").textContent = data.error || "Не удалось сохранить шаблон."; return; }
  $("templateModal").hidden = true;
  await loadTemplates();
  $("templateSelect").value = data.id;
  state.templateId = data.id;
}

async function deleteTemplateConfirm() {
  if (!state.editingTemplateId) { $("templateModal").hidden = true; return; }
  if (!confirm("Удалить этот шаблон анкеты?")) return;
  const r = await fetch(`${HTTP}/api/templates/${state.editingTemplateId}`, { method: "DELETE" });
  const data = await r.json();
  if (!r.ok) { $("templateHint").textContent = data.error || "Не удалось удалить шаблон."; return; }
  $("templateModal").hidden = true;
  await loadTemplates();
}

async function confirmAnswer() {
  hideAutoBoxes();
  stopAutoListening();
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
    hideAutoBoxes();
    stopAutoListening();   // ручная запись — приоритет над автослушанием, если оно ещё идёт
    answerRec = new window.AudioCapture.OneShotRecorder(answerDeviceId());
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

// ---------- Протокол: стриминг (Блок 2 — partial/stable/final) ----------
function startRecording() {
  state.ws = new WebSocket(`${WS}/ws/stream/${SESSION}`);
  state.ws.binaryType = "arraybuffer";
  state.ws.onopen = async () => {
    // Сессия создаётся сервером автоматически при подключении к сокету — отдельный
    // "start"-сигнал не нужен (сервер его и не обрабатывает).
    if ($("stereoSplit").checked) {
      // Оба микрофона в одном адаптере: расщепляем его стерео на каналы 0 и 1.
      const cap = new window.AudioCapture.StereoSplitCapture($("mic0").value, state.ws);
      try {
        await cap.start();
      } catch (e) {
        alert("Стерео-режим не вышел: " + e.message +
              "\nСнимите галочку «стерео-вход» либо проверьте адаптер: py -3.11 -m app.test_mics");
        state.ws.close();
        return;
      }
      state.captures.push(cap);
    } else {
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
    }
    state.recording = true;
    state.flushed = true;
    $("startRec").disabled = true;
    $("stopRec").disabled = false;
    updateModelSelectEnabled();
  };
  state.ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "asr_update" || msg.type === "asr_partial") {
      updateLiveSegment(msg);
    } else if (msg.type === "asr_final") {
      finalizeLiveSegment(msg);
    } else if (msg.type === "segment_update") {
      applyBleedFlag(msg);
    } else if (msg.type === "stopped") {
      state.flushed = true;
      $("finalizeBtn").disabled = false;
      $("diarizeBtn").disabled = false;
      $("saveBtn").disabled = false;
    }
  };
}

function stopRecording() {
  state.captures.forEach((c) => c.stop());
  state.captures = [];
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    // До прихода "stopped" хвостовые реплики ещё не гарантированно долетели — блокируем
    // финализацию/сохранение, чтобы не потерять последние секунды разговора в протоколе.
    state.flushed = false;
    $("finalizeBtn").disabled = true;
    $("diarizeBtn").disabled = true;
    $("saveBtn").disabled = true;
    state.ws.send(JSON.stringify({ type: "stop" }));
  }
  state.recording = false;
  $("startRec").disabled = false;
  $("stopRec").disabled = true;
  updateModelSelectEnabled();
}

function speakerClass(ch) {
  return ch === 0 ? "ch0" : ch === 1 ? "ch1" : "auto";
}

// ---------- "живой" сегмент: partial/update (ещё не финализирован) ----------
function updateLiveSegment(msg) {
  const key = String(msg.channel);
  let live = state.liveSegments.get(key);
  if (!live || live.utteranceId !== msg.utterance_id) {
    if (live) live.el.remove();   // прошлая реплика этого канала не дождалась финала — убираем
    live = createLiveSegment(msg.channel, msg.utterance_id);
    state.liveSegments.set(key, live);
  }
  renderLiveText(live, msg.text || "", msg.stable_word_count || 0);
}

function createLiveSegment(channel, utteranceId) {
  const wrap = document.createElement("div");
  wrap.className = "segment segment-live";
  const sp = document.createElement("div");
  sp.className = "seg-speaker " + speakerClass(channel);
  sp.textContent = "…";
  const body = document.createElement("div");
  body.className = "seg-body";
  const txt = document.createElement("div");
  txt.className = "seg-text";
  body.appendChild(txt);
  wrap.appendChild(sp);
  wrap.appendChild(body);
  $("transcript").appendChild(wrap);
  $("transcript").scrollTop = $("transcript").scrollHeight;
  return { el: wrap, txtEl: txt, channel, utteranceId };
}

function renderLiveText(live, text, stableWordCount) {
  live.txtEl.innerHTML = "";
  text.split(/\s+/).filter(Boolean).forEach((w, i) => {
    const span = document.createElement("span");
    span.className = "word " + (i < stableWordCount ? "stable" : "fluid");
    span.textContent = (i > 0 ? " " : "") + w;
    live.txtEl.appendChild(span);
  });
  $("transcript").scrollTop = $("transcript").scrollHeight;
}

function finalizeLiveSegment(msg) {
  const key = String(msg.channel);
  const live = state.liveSegments.get(key);
  if (live && live.utteranceId === msg.utterance_id) {
    live.el.remove();
    state.liveSegments.delete(key);
  }
  if (msg.text && msg.id) addSegment(msg);
}

function addSegment(seg) {
  const wrap = document.createElement("div");
  wrap.className = "segment";
  wrap.dataset.id = seg.id;

  const sp = document.createElement("div");
  sp.className = "seg-speaker " + speakerClass(seg.channel);
  sp.textContent = seg.speaker + ":";
  sp.title = "Клик — изменить метку голоса";
  sp.onclick = () => openSpeakerModal(seg.channel, seg.speaker);

  if (seg.likely_bleed) {
    wrap.classList.add("segment-bleed");
    const flag = document.createElement("span");
    flag.className = "bleed-flag";
    flag.title = "Похоже на протёкший голос другого канала (совпадает с репликой на другом " +
      "канале примерно в то же время) — авто-решение не принято, проверьте вручную.";
    flag.textContent = "⚠ вероятный дубль";
    flag.onclick = (ev) => { ev.stopPropagation(); wrap.classList.toggle("collapsed"); };
    sp.appendChild(document.createElement("br"));
    sp.appendChild(flag);
  }

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
  if (seg.aligned) {
    const tag = document.createElement("span");
    tag.textContent = "  · ✓ тайм-коды уточнены";
    tag.style.color = "#2f8a57";
    meta.appendChild(tag);
  }

  body.appendChild(txt);
  body.appendChild(meta);
  wrap.appendChild(sp);
  wrap.appendChild(body);
  $("transcript").appendChild(wrap);
  $("transcript").scrollTop = $("transcript").scrollHeight;

  // speaker храним: переименование идёт ПО МЕТКЕ голоса (после диаризации в одном канале
  // их несколько), и без неё нечего сравнивать при локальном обновлении ленты.
  state.segments.set(seg.id, { el: wrap, speakerEl: sp, channel: seg.channel,
                               speaker: seg.speaker, words, meta });
}

function applyBleedFlag(seg) {
  // Сегмент уже был отрисован раньше как обычный, но пост-ASR дедупликация (Блок 3.7) задним
  // числом распознала его как вероятный дубль протёкшего голоса — досвечиваем на месте.
  const s = state.segments.get(seg.id);
  if (!s || s.el.classList.contains("segment-bleed")) return;
  s.el.classList.add("segment-bleed");
  const flag = document.createElement("span");
  flag.className = "bleed-flag";
  flag.title = "Похоже на протёкший голос другого канала — проверьте вручную.";
  flag.textContent = "⚠ вероятный дубль";
  flag.onclick = (ev) => { ev.stopPropagation(); s.el.classList.toggle("collapsed"); };
  s.speakerEl.appendChild(document.createElement("br"));
  s.speakerEl.appendChild(flag);
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

// ---------- Финализация: уточнение тайм-кодов (forced alignment) ----------
function renderProtocol(protocol) {
  $("transcript").innerHTML = "";
  state.segments.clear();
  (protocol.segments || []).forEach(addSegment);
}

// Разметка голосов отдельной кнопкой, а не галочкой при «Уточнить тайм-коды». Это разные
// операции с разной ценой и разным смыслом: тайм-коды уточняются всегда и никого не
// переименовывают, а диаризация нужна только в режиме общего микрофона и переписывает метки
// спикеров. Спрятанная в чекбокс, она была попросту незаметна.
async function diarizeVoices() {
  const btn = $("diarizeBtn");
  const prev = btn.textContent;
  btn.disabled = true;
  btn.textContent = "⏳ Разметка…";
  const num = parseInt($("numSpeakers").value, 10) || 0;
  try {
    const r = await fetch(`${HTTP}/api/finalize`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      // align здесь НЕ трогаем: уточнение тайм-кодов — отдельная кнопка, и падение тяжёлой
      // wav2vec2-модели не должно уносить с собой уже посчитанную разметку голосов.
      body: JSON.stringify({
        session_id: SESSION, align: false, diarize: true,
        num_speakers: num > 0 ? num : null,
      }),
    }).then((x) => x.json());
    if (r.error) {
      alert("Не удалось разметить голоса: " + r.error);
      return;
    }
    renderProtocol(r.protocol);
    alert(`Готово — голосов найдено: ${r.speakers ?? "?"}.
` +
          "Метки «Голос-N» переименовываются кликом по имени слева от реплики.");
  } catch (e) {
    alert("Ошибка: " + e);
  } finally {
    btn.disabled = false;
    btn.textContent = prev;
  }
}

async function finalizeTimecodes() {
  const btn = $("finalizeBtn");
  btn.disabled = true;
  const prev = btn.textContent;
  btn.textContent = "⏳ Обработка…";
  const num = parseInt($("numSpeakers").value, 10) || 0;
  try {
    const r = await fetch(`${HTTP}/api/finalize`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: SESSION, align: true, diarize: false,
        num_speakers: num > 0 ? num : null,
      }),
    }).then((x) => x.json());
    if (r.error) {
      alert("Ошибка финализации: " + r.error);
    } else {
      renderProtocol(r.protocol);
      await loadAudio();
      alert(`Готово — тайм-коды уточнены: ${r.aligned_segments ?? 0} сегм.`);
    }
  } catch (e) {
    alert("Ошибка: " + e);
  } finally {
    btn.disabled = false;
    btn.textContent = prev;
  }
}

// ---------- Маркировка спикеров ----------
let modalChannel = null;
let modalSpeaker = null;   // текущая метка голоса, по которой и переименовываем
function openSpeakerModal(channel, speaker) {
  modalChannel = channel;
  modalSpeaker = speaker || null;
  $("modalChannel").textContent = speaker ? `«${speaker}»` : `канал ${channel}`;
  $("speakerInput").value = "";
  $("speakerModal").hidden = false;
}
async function applySpeaker(label) {
  if (!label) label = $("speakerInput").value.trim();
  if (!label || modalChannel === null) return;
  // Лента перерисовывается локально, поэтому ответ обязателен к проверке: на истёкшей сессии
  // /api/speaker отдаёт 404 «no session», сервер ничего не переименовал — и молчаливая
  // перерисовка показала бы новую метку, которой в протоколе нет.
  let r;
  try {
    r = await fetch(`${HTTP}/api/speaker`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      // speaker — какой именно голос переименовываем. После диаризации общего микрофона в
      // одном канале лежат разные спикеры, и переименование «по каналу» схлопнуло бы весь
      // протокол в одну метку (реальный баг: «Голос-2» -> «Опрашивающий» у всех реплик).
      body: JSON.stringify({ session_id: SESSION, channel: modalChannel, label,
                             speaker: modalSpeaker }),
    }).then((x) => x.json());
  } catch (e) {
    alert("Ошибка: " + e);
    return;
  }
  if (r.error) {
    alert("Не удалось переименовать голос: " + r.error);
    return;
  }
  // Локально обновляем ленту, не перезагружая протокол.
  state.segments.forEach((s) => {
    // Обновляем ровно те строки, что реально переименованы на сервере.
    const wasLabel = modalSpeaker;
    if (wasLabel) {
      if (s.speaker === wasLabel) { s.speaker = label; s.speakerEl.textContent = label + ":"; }
    } else if (s.channel === modalChannel) {
      s.speaker = label;
      s.speakerEl.textContent = label + ":";
    }
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
  $("finalizeBtn").onclick = finalizeTimecodes;
  $("diarizeBtn").onclick = diarizeVoices;
  $("saveBtn").onclick = () =>
    fetch(`${HTTP}/api/save`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION }),
    }).then(() => alert("Протокол сохранён на сервере."));
  $("loadAudio").onclick = loadAudio;
  $("refreshMics").onclick = loadMics;
  $("mic0").onchange = onMicChosen;
  $("mic1").onchange = onMicChosen;
  $("stereoSplit").onchange = checkMicsDistinct;
  $("modelSelect").onchange = onModelChange;
  $("audio").addEventListener("timeupdate", onTimeUpdate);

  $("speakerCancel").onclick = () => ($("speakerModal").hidden = true);
  $("speakerApply").onclick = () => applySpeaker();
  document.querySelectorAll("#speakerModal .presets button").forEach((b) => {
    b.onclick = () => applySpeaker(b.dataset.v);
  });

  $("templateSelect").onchange = () => { state.templateId = $("templateSelect").value || null; };
  $("newTemplateBtn").onclick = () => openTemplateEditor(null);
  $("editTemplateBtn").onclick = () => {
    if (state.templateId) openTemplateEditor(state.templateId);
  };
  $("addStepBtn").onclick = () => {
    $("templateSteps").appendChild(stepRowTemplate({ kind: "field" }));
  };
  $("templateCancel").onclick = () => ($("templateModal").hidden = true);
  $("templateSaveBtn").onclick = saveTemplate;
  $("templateDeleteBtn").onclick = deleteTemplateConfirm;
}

bind();
checkHealth();
loadMics();
loadModels();
loadTemplates();
