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
  // Импорт .docx-бланка (Блок 6): id/имя докс-файла черновика, которые нужно передать при
  // сохранении шаблона (POST /api/templates), иначе сохранённый шаблон получит другой id, и
  // уже нормализованный докс-файл на сервере останется ни на что не сославшимся.
  importedDocxId: null,
  importedDocxFilename: null,
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

  // source="manual" (Блок 6) — оператор вводит значение сам с клавиатуры, это НЕ голосовой
  // ответ опрашиваемого. Раньше такой шаг всё равно проходил через озвучку + автослушание —
  // ассистент либо молчал (если formulировка не заполнена — see Step.prompt на бэкенде), либо
  // впустую слушал 12 секунд тишины, что выглядело как «шаг вообще пропущен» (жалоба
  // пользователя). Для manual сразу показываем поле ввода, без TTS и без микрофона.
  if (step.source === "manual") {
    if (!step.needs_answer) { advanceInfo(); return; }   // нечего вводить — как и в голосовом потоке
    $("answerInput").focus();
    return;
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
  const r = await fetch(`${HTTP}/api/transcribe?session_id=${encodeURIComponent(SESSION)}`, {
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
      o.dataset.builtin = t.is_builtin ? "1" : "";
      sel.appendChild(o);
    });
    sel.value = list.some((t) => t.id === prev) ? prev : (list[0] ? list[0].id : "");
    state.templateId = sel.value || null;
    updateDeleteTemplateBtnState();
  } catch {}
}

// Стандартный шаблон не удаляется (см. TemplateStore.delete в backend) — вместо того чтобы
// давать нажать и упереться в алерт с ошибкой, сразу выключаем кнопку.
function updateDeleteTemplateBtnState() {
  const opt = $("templateSelect").selectedOptions[0];
  $("deleteSelectedTemplateBtn").disabled = !opt || opt.dataset.builtin === "1";
}

let draggedStepRow = null;   // текущий перетаскиваемый .step-row (один редактор шаблона зараз)

function stepRowTemplate(step) {
  const row = document.createElement("div");
  row.className = "step-row";
  row.innerHTML = `
    <div class="step-head">
      <span class="step-drag" draggable="true" title="Перетащите, чтобы изменить порядок">⠿</span>
      <input class="step-label" type="text" placeholder="Название шага" value="${escAttr(step.label || "")}" />
      <div class="step-order">
        <button type="button" class="step-up" title="Выше">▲</button>
        <button type="button" class="step-down" title="Ниже">▼</button>
      </div>
      <button type="button" class="step-remove" title="Удалить шаг">✕</button>
    </div>
    <div class="step-meta">
      <label>Тип
        <select class="step-kind" title="«Только зачитать» — ответ вообще не запрашивается и не сохраняется (для разъяснений прав и т.п.). Для полей с данными, даже вводимых вручную, нужен «вопрос с ответом».">
          <option value="field">вопрос с ответом</option>
          <option value="confirm">да/нет</option>
          <option value="info">только зачитать (без ответа)</option>
        </select>
      </label>
      <label class="step-extractor-label">Извлечение
        <select class="step-extractor">
          <option value="plain">как есть</option>
          <option value="fio">ФИО</option>
          <option value="birth">дата</option>
          <option value="yesno">да/нет</option>
          <option value="none">не сохранять</option>
        </select>
      </label>
      <label class="step-auto-kind-label" hidden>Авто-поле
        <select class="step-auto-kind">
          <option value="date">сегодняшняя дата</option>
          <option value="time_start">время начала опроса</option>
          <option value="time_end">время окончания опроса</option>
          <option value="time_range">время начала и окончания одной строкой</option>
        </select>
      </label>
      <label>Источник
        <select class="step-source">
          <option value="asr">голосом (ASR)</option>
          <option value="manual">текстом вручную (каждый раз)</option>
          <option value="profile">из профиля оператора (один раз навсегда)</option>
          <option value="auto">автоматически (дата/время опроса)</option>
        </select>
      </label>
      <label>Плейсхолдер .docx
        <input class="step-placeholder" type="text" placeholder="напр. T1.PARTICIP_SURNAME"
               value="${escAttr(step.placeholder || "")}" />
      </label>
    </div>
    <div class="step-texts">
      <textarea class="step-statement" placeholder="Текст для зачитывания (разъяснение, опционально)">${step.statement || ""}</textarea>
      <textarea class="step-question" placeholder="Вопрос (если нужен ответ)">${step.question || ""}</textarea>
    </div>
  `;
  row.querySelector(".step-kind").value = step.kind || "field";
  const AUTO_KINDS = ["date", "time_start", "time_end", "time_range"];
  const isAuto = step.source === "auto";
  row.querySelector(".step-extractor").value = (!isAuto && step.extractor) || "plain";
  row.querySelector(".step-auto-kind").value = (isAuto && AUTO_KINDS.includes(step.extractor))
    ? step.extractor : "date";
  row.querySelector(".step-source").value = step.source || "asr";
  // Извлечение (ASR) и Авто-поле — два разных смысла одного и того же значения `extractor`
  // (см. models.py::TemplateStep.source), одновременно оба показывать незачем — путает, какое
  // из них сейчас реально используется.
  const toggleAutoUi = () => {
    const auto = row.querySelector(".step-source").value === "auto";
    row.querySelector(".step-extractor-label").hidden = auto;
    row.querySelector(".step-auto-kind-label").hidden = !auto;
  };
  toggleAutoUi();
  row.querySelector(".step-source").onchange = toggleAutoUi;
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

  // Drag-and-drop реордер (в дополнение к ▲/▼ — тем удобнее менять порядок на одну позицию,
  // этим на несколько сразу). Хендл — отдельный элемент, а не вся строка: иначе перетаскивание
  // текста в поле «Название шага» (выделение мышью) конфликтовало бы с перетаскиванием шага.
  const handle = row.querySelector(".step-drag");
  handle.addEventListener("dragstart", (e) => {
    draggedStepRow = row;
    row.classList.add("dragging");
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", "");   // Firefox не начинает drag без данных
    startStepDragAutoScroll(row.closest(".modal-card-wide"));
  });
  handle.addEventListener("dragend", () => {
    row.classList.remove("dragging");
    draggedStepRow = null;
    stopStepDragAutoScroll();
  });
  row.addEventListener("dragover", (e) => {
    if (!draggedStepRow || draggedStepRow === row) return;
    e.preventDefault();   // разрешить drop именно сюда
    dragAutoScrollPointerY = e.clientY;   // автоскролл читает это на каждом кадре, см. ниже
    const rect = row.getBoundingClientRect();
    const before = e.clientY - rect.top < rect.height / 2;
    row.parentNode.insertBefore(draggedStepRow, before ? row : row.nextSibling);
  });
  row.addEventListener("drop", (e) => e.preventDefault());
  return row;
}

// Автоскролл списка шагов во время drag-and-drop (Блок 6): нативный HTML5 DnD автоскроллит
// только document/window, а список шагов скроллится ВНУТРИ модалки (.modal-card-wide) — без
// этого перетащить шаг из конца длинного списка в начало было физически невозможно (жалоба
// пользователя). requestAnimationFrame, а не сам dragover — dragover у неподвижного курсора
// срабатывает нерегулярно (раз в ~350мс по спеке), рывками, а не плавно.
let dragAutoScrollPointerY = null;
let dragAutoScrollHandle = null;

function startStepDragAutoScroll(container) {
  if (!container) return;
  const EDGE = 60, MAX_SPEED = 16;
  const tick = () => {
    if (!draggedStepRow) { dragAutoScrollHandle = null; return; }   // drag уже закончен
    if (dragAutoScrollPointerY !== null) {
      const rect = container.getBoundingClientRect();
      let delta = 0;
      if (dragAutoScrollPointerY < rect.top + EDGE) {
        delta = -MAX_SPEED * (1 - (dragAutoScrollPointerY - rect.top) / EDGE);
      } else if (dragAutoScrollPointerY > rect.bottom - EDGE) {
        delta = MAX_SPEED * (1 - (rect.bottom - dragAutoScrollPointerY) / EDGE);
      }
      if (delta) container.scrollTop += delta;
    }
    dragAutoScrollHandle = requestAnimationFrame(tick);
  };
  dragAutoScrollHandle = requestAnimationFrame(tick);
}

function stopStepDragAutoScroll() {
  if (dragAutoScrollHandle) cancelAnimationFrame(dragAutoScrollHandle);
  dragAutoScrollHandle = null;
  dragAutoScrollPointerY = null;
}

function escAttr(s) {
  return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

// ---------- профиль оператора (Блок 6) ----------
async function openProfileModal() {
  $("profileHint").textContent = "";
  const box = $("profileFields");
  box.innerHTML = "";
  const [fields, values] = await Promise.all([
    fetch(`${HTTP}/api/profile/fields`).then((x) => x.json()),
    fetch(`${HTTP}/api/profile`).then((x) => x.json()),
  ]);
  if (!fields.length) {
    box.innerHTML = '<p class="hint">Пока нет ни одного поля с источником «из профиля '
      + 'оператора» ни в одном шаблоне — добавьте его в редакторе шаблона (Источник → '
      + '«из профиля оператора»).</p>';
  } else {
    fields.forEach((f) => {
      const row = document.createElement("label");
      row.style.display = "block";
      row.style.marginBottom = "10px";
      row.innerHTML = `${escAttr(f.label)}
        <input type="text" data-placeholder="${escAttr(f.placeholder)}"
               value="${escAttr(values[f.placeholder] || "")}" />`;
      box.appendChild(row);
    });
  }
  $("profileModal").hidden = false;
}

async function saveProfile() {
  const values = {};
  document.querySelectorAll("#profileFields input").forEach((i) => {
    values[i.dataset.placeholder] = i.value.trim();
  });
  const r = await fetch(`${HTTP}/api/profile`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values }),
  });
  if (!r.ok) { $("profileHint").textContent = "Не удалось сохранить профиль."; return; }
  $("profileModal").hidden = true;
}

async function openTemplateEditor(templateId) {
  state.editingTemplateId = templateId;
  state.importedDocxId = null;
  state.importedDocxFilename = null;
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
  // Существующий шаблон уже привязан к своему докс-файлу на сервере — сохраняем ссылку на него,
  // чтобы PUT-редактирование не потеряло привязку (docx_filename не восстанавливается сам по себе).
  state.importedDocxFilename = tmpl.docx_filename || null;
  (tmpl.steps || []).forEach((s) => stepsBox.appendChild(stepRowTemplate(s)));
  $("templateQaPlaceholder").value = tmpl.qa_placeholder || "";
  refreshQaPlaceholderRequired();
  $("templateDeleteBtn").hidden = !state.editingTemplateId;
  $("templateModal").hidden = false;
}

async function importDocxTemplate(file) {
  if (!file) return;
  // Если в редакторе уже открыт шаблон (создавали ли его импортом раньше или вручную) — это
  // ПЕРЕимпорт того же протокола (например, чтобы подхватить исправленный .docx или заново
  // просканировать плейсхолдеры после починки багов), а не создание нового с нуля. Название,
  // формулировки, источники и т.п. уже настроены руками — переносить их заново не хочется
  // (жалоба пользователя). Ловим ДО того, как перезапишем templateSteps черновиком.
  const isReimport = $("templateSteps").querySelectorAll(".step-row").length > 0;
  const oldSteps = isReimport ? collectStepsFromEditor() : [];
  const oldByPlaceholder = new Map(oldSteps.filter((s) => s.placeholder).map((s) => [s.placeholder, s]));
  const oldWithoutPlaceholder = oldSteps.filter((s) => !s.placeholder);   // свои шаги вне докса

  $("templateHint").textContent = "Импортирую .docx…";
  const form = new FormData();
  form.append("file", file);
  let draft;
  try {
    const r = await fetch(`${HTTP}/api/templates/import_docx`, { method: "POST", body: form });
    draft = await r.json();
    if (!r.ok) { $("templateHint").textContent = draft.error || "Не удалось импортировать .docx."; return; }
  } catch (e) {
    $("templateHint").textContent = "Ошибка импорта: " + e;
    return;
  }
  // Черновик из /api/templates/import_docx не сохранён в хранилище (см. main.py) — это только
  // свежий докс-файл + список найденных в нём токенов. Дальше для каждого токена подставляем
  // УЖЕ настроенный шаг, если он был (по совпадению placeholder), иначе — заготовку по умолчанию.
  const mergedSteps = (draft.steps || []).map((s) => oldByPlaceholder.get(s.placeholder) || s);
  const newCount = mergedSteps.filter((s) => !oldByPlaceholder.has(s.placeholder)).length;

  state.importedDocxId = draft.id;
  state.importedDocxFilename = draft.docx_filename;
  if (!isReimport) {
    // По-настоящему новый шаблон с нуля — раньше это был единственный сценарий импорта.
    state.editingTemplateId = null;
    $("templateName").value = "";
    $("templateDescription").value = "";
    $("templateQaPlaceholder").value = "";
    $("templateModalTitle").textContent = "Новый шаблон из .docx";
    $("templateDeleteBtn").hidden = true;
  }
  // state.editingTemplateId, название, описание и qa_placeholder при переимпорте НЕ трогаем —
  // сохранение (PUT) обновит тот же шаблон на месте с новым docx_filename и слитыми шагами.
  const stepsBox = $("templateSteps");
  stepsBox.innerHTML = "";
  mergedSteps.forEach((s) => stepsBox.appendChild(stepRowTemplate(s)));
  oldWithoutPlaceholder.forEach((s) => stepsBox.appendChild(stepRowTemplate(s)));
  refreshQaPlaceholderRequired();
  $("templateHint").textContent = isReimport
    ? `Переимпорт: ${mergedSteps.length} полей найдено в .docx, из них новых — ${newCount}. ` +
      `Настройки уже существующих полей сохранены. Проверьте новые поля и сохраните.`
    : `Найдено полей: ${mergedSteps.length}. Заполните название, при необходимости — ` +
      `вопросы/формулировки для каждого поля, и плейсхолдер для протокола диалога, затем сохраните.`;
}

// Плейсхолдер «Протокол диалога» — обязателен только когда к шаблону привязан .docx: у чисто
// голосовых анкет (без докса) поле ни на что не влияет и не должно блокировать сохранение.
function refreshQaPlaceholderRequired() {
  const required = !!state.importedDocxFilename;
  $("templateQaPlaceholder").required = required;
  $("templateQaPlaceholderHint").classList.toggle("required", required);
  $("templateQaPlaceholderHint").textContent = required
    ? "Обязательно: без него записанный диалог не попадёт в итоговый документ."
    : "Впишите, если хотите вставлять в документ записанный диалог (актуально только для шаблонов с .docx).";
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
    const source = row.querySelector(".step-source").value;
    return {
      key, label,
      kind: row.querySelector(".step-kind").value,
      // Для source="auto" `extractor` — не имя ASR-извлекателя, а выбранное авто-поле
      // (см. models.py::TemplateStep.source и docgen.AUTO_FIELDS).
      extractor: source === "auto"
        ? row.querySelector(".step-auto-kind").value
        : row.querySelector(".step-extractor").value,
      source,
      placeholder: row.querySelector(".step-placeholder").value.trim(),
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
  const qaPlaceholder = $("templateQaPlaceholder").value.trim() || null;
  // Обязательно у шаблонов с докс-файлом — иначе записанный диалог молча не попадёт в
  // итоговый документ (docgen.render просто не знает, в какой плейсхолдер его вставлять).
  if (state.importedDocxFilename && !qaPlaceholder) {
    $("templateHint").textContent =
      "Впишите плейсхолдер для стенограммы «Протокол диалога» — без него записанный диалог не попадёт в документ.";
    return;
  }
  // «Только зачитать» (kind=info) — это НЕ «поле без специальной обработки»: ответ на такой шаг
  // вообще не запрашивается и не сохраняется (см. AssistantSession.to_fields/submit_answer), так
  // что привязанный к нему плейсхолдер никогда не заполнится. Реальная ошибка пользователя —
  // спутал с «Извлечение → как есть». Предупреждаем, но не блокируем — мало ли реальный случай.
  const infoWithPlaceholder = steps.filter((s) => s.kind === "info" && s.placeholder);
  if (infoWithPlaceholder.length) {
    const list = infoWithPlaceholder.map((s) => `«${s.label}» (${s.placeholder})`).join(", ");
    if (!confirm(
      `У шагов с типом «только зачитать» указан плейсхолдер — их значение НИКОГДА не сохранится ` +
      `(ответ на такой шаг вообще не запрашивается): ${list}.\n\n` +
      `Если это поле должно заполняться (оператором вручную или голосом), смените Тип на ` +
      `«вопрос с ответом».\n\nВсё равно сохранить как есть?`
    )) return;
  }
  const payload = {
    name, description: $("templateDescription").value.trim(), steps,
    docx_filename: state.importedDocxFilename || null,
    qa_placeholder: qaPlaceholder,
  };
  // id значим только при создании нового шаблона из импортированного .docx-черновика (см.
  // importDocxTemplate) — без него сохранённый шаблон получил бы другой id, и уже
  // нормализованный докс-файл на сервере остался бы ни на что не сославшимся (main.py комментарий).
  if (!state.editingTemplateId && state.importedDocxId) payload.id = state.importedDocxId;
  const body = JSON.stringify(payload);
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
  await warnAboutUncoveredPlaceholders(data.id);
}

// Плейсхолдеры бланка без единого привязанного шага (Блок 6) — реальный случай: при
// редактировании шаблона шаг удалили или переименовали плейсхолдер с опечаткой, и такое поле в
// итоговом документе тихо остаётся пустым, а голосовая анкета вообще не спросит нужные данные.
// Не блокирует сохранение — иногда часть плейсхолдеров бланка оставляют незаполненной осознанно.
async function warnAboutUncoveredPlaceholders(templateId) {
  try {
    const r = await fetch(`${HTTP}/api/templates/${templateId}/coverage`);
    const data = await r.json();
    if (data.missing && data.missing.length) {
      alert("В .docx-бланке есть поля без привязанного шага анкеты — они останутся пустыми в "
        + "документе:\n\n" + data.missing.join(", ")
        + "\n\nДобавьте для них шаги в редакторе шаблона.");
    }
  } catch {}
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

// Удаление прямо из выбора шаблона (Блок 6) — без открытия полного редактора: раньше
// единственный способ удалить шаблон был через «✎ Редактировать» -> прокрутить длинный список
// шагов до кнопки внизу модалки, что для простого «убрать ненужный шаблон» избыточно.
async function deleteSelectedTemplate() {
  if (!state.templateId) return;
  const label = $("templateSelect").selectedOptions[0]?.textContent || "выбранный шаблон";
  if (!confirm(`Удалить шаблон «${label}»? Это необратимо.`)) return;
  const r = await fetch(`${HTTP}/api/templates/${state.templateId}`, { method: "DELETE" });
  const data = await r.json();
  if (!r.ok) { alert(data.error || "Не удалось удалить шаблон."); return; }
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
    const r = await fetch(`${HTTP}/api/transcribe?session_id=${encodeURIComponent(SESSION)}`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" },
      body: pcm.buffer,
    }).then((x) => x.json());
    $("answerInput").value = r.text || "";
  }
}

// Значение поля редактируется прямо в таблице (Блок 6): окно автоподтверждения голосового
// ответа висит пару секунд — оператор часто не успевает нажать «исправить», а поля с
// source="manual" вообще не задумывались как голосовые — таблица должна быть основным способом
// их заполнения, не только резервной правкой.
function renderFields(fields) {
  const tb = $("fieldsTable").querySelector("tbody");
  tb.innerHTML = "";
  (fields || []).forEach((f) => {
    const tr = document.createElement("tr");
    const k = document.createElement("td");
    k.className = "k";
    k.textContent = f.label;
    const v = document.createElement("td");
    v.className = "v" + (f.value ? " filled" : "");
    v.contentEditable = "true";
    v.textContent = f.value || "";
    v.dataset.placeholder = "—";   // см. CSS: пустая ячейка показывает плейсхолдер, не теряя редактируемость
    v.addEventListener("blur", () => saveFieldEdit(f.key, v));
    tr.appendChild(k);
    tr.appendChild(v);
    tb.appendChild(tr);
  });
}

async function saveFieldEdit(key, td) {
  const value = td.innerText.trim();
  const r = await fetch(`${HTTP}/api/assistant/field`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION, key, value }),
  }).then((x) => x.json());
  if (!r.ok) return;   // ключа нет в текущем сценарии — сюда в норме не попасть, но не рвём UI
  td.classList.toggle("filled", !!value);
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

// ---------- генерация .docx-протокола (Блок 6) ----------
async function generateProtocolDocx() {
  const btn = $("genDocxBtn");
  const prev = btn.textContent;
  btn.disabled = true;
  btn.textContent = "⏳ Формирую…";
  try {
    const r = await fetch(`${HTTP}/api/protocol/${SESSION}/docx`, { method: "POST" });
    if (!r.ok) {
      let msg = "Не удалось сформировать протокол.";
      try { msg = (await r.json()).error || msg; } catch {}
      alert(msg);
      return;
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `protocol_${SESSION}.docx`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert("Ошибка: " + e);
  } finally {
    btn.disabled = false;
    btn.textContent = prev;
  }
}

// ---------- клик вне модалки закрывает её ----------
function initModalBackdropClose() {
  // "click" одним условием (target === modal) ловит не только клик по фону, но и выделение
  // текста в узком поле внутри карточки: пока тянешь мышь, курсор легко уезжает за пределы
  // инпута на фон модалки, и click-событие засчитывается по фону — модалка закрывалась прямо
  // посреди выделения (жалоба пользователя). Значит закрывать нужно, только если И нажатие,
  // И отпускание мыши пришлись именно на фон, а не по итоговой точке составного клика.
  document.querySelectorAll(".modal").forEach((modal) => {
    let downOnBackdrop = false;
    modal.addEventListener("mousedown", (e) => { downOnBackdrop = e.target === modal; });
    modal.addEventListener("mouseup", (e) => {
      if (downOnBackdrop && e.target === modal) modal.hidden = true;
      downOnBackdrop = false;
    });
  });
}

// ---------- перетаскиваемая граница между анкетой и протоколом диалога ----------
const SPLITTER_MIN = 300;      // ужать анкету настолько, чтобы поля ещё влезали без переноса
const SPLITTER_STORAGE_KEY = "protocolAssistant.leftWidth";

function initSplitter() {
  const saved = Number(localStorage.getItem(SPLITTER_STORAGE_KEY));
  if (saved) document.documentElement.style.setProperty("--left-width", saved + "px");

  const splitter = $("splitter");
  const main = document.querySelector("main");
  splitter.addEventListener("mousedown", (e) => {
    e.preventDefault();
    splitter.classList.add("dragging");
    const onMove = (ev) => {
      const rect = main.getBoundingClientRect();
      const max = rect.width - SPLITTER_MIN;   // правой колонке тоже нужен минимум места
      const w = Math.min(Math.max(ev.clientX - rect.left, SPLITTER_MIN), max);
      document.documentElement.style.setProperty("--left-width", w + "px");
    };
    const onUp = () => {
      splitter.classList.remove("dragging");
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      const w = getComputedStyle(document.documentElement).getPropertyValue("--left-width");
      if (w) localStorage.setItem(SPLITTER_STORAGE_KEY, parseInt(w, 10));
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
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
  $("genDocxBtn").onclick = generateProtocolDocx;
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

  $("operatorProfileBtn").onclick = openProfileModal;
  $("profileCancel").onclick = () => ($("profileModal").hidden = true);
  $("profileSaveBtn").onclick = saveProfile;

  $("templateSelect").onchange = () => {
    state.templateId = $("templateSelect").value || null;
    updateDeleteTemplateBtnState();
  };
  $("newTemplateBtn").onclick = () => openTemplateEditor(null);
  $("editTemplateBtn").onclick = () => {
    if (state.templateId) openTemplateEditor(state.templateId);
  };
  $("deleteSelectedTemplateBtn").onclick = deleteSelectedTemplate;
  $("addStepBtn").onclick = () => {
    $("templateSteps").appendChild(stepRowTemplate({ kind: "field" }));
  };
  $("templateCancel").onclick = () => ($("templateModal").hidden = true);
  $("templateSaveBtn").onclick = saveTemplate;
  $("templateDeleteBtn").onclick = deleteTemplateConfirm;

  $("importDocxBtn").onclick = () => $("importDocxInput").click();
  $("importDocxInput").onchange = async (e) => {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";   // сброс, иначе повторный выбор того же файла не вызовет onchange
    if (file) await importDocxTemplate(file);
  };
}

bind();
initSplitter();
initModalBackdropClose();
checkHealth();
loadMics();
loadModels();
loadTemplates();
