// Захват нескольких микрофонов и отправка PCM (int16, 16кГц) по WebSocket.
// Каждый кадр: [int32 LE channel][int16 LE samples...].
// Канал = индекс микрофона = спикер (диаризация по каналу).

const TARGET_RATE = 16000;

function floatTo16BitPCM(input) {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

// Линейный даунсемпл из sourceRate в TARGET_RATE.
function downsample(buffer, sourceRate) {
  if (sourceRate === TARGET_RATE) return buffer;
  const ratio = sourceRate / TARGET_RATE;
  const newLen = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLen);
  let pos = 0;
  for (let i = 0; i < newLen; i++) {
    const next = Math.round((i + 1) * ratio);
    let sum = 0, count = 0;
    for (let j = Math.round(i * ratio); j < next && j < buffer.length; j++) {
      sum += buffer[j];
      count++;
    }
    result[i] = count ? sum / count : 0;
    pos = next;
  }
  return result;
}

class CaptureChannel {
  constructor(channel, deviceId, ws) {
    this.channel = channel;
    this.deviceId = deviceId;
    this.ws = ws;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: this.deviceId ? { exact: this.deviceId } : undefined,
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 1,
      },
    });
    this.ctx = new AudioContext();
    const source = this.ctx.createMediaStreamSource(this.stream);
    // ScriptProcessor — простой путь для MVP.
    this.node = this.ctx.createScriptProcessor(4096, 1, 1);
    const header = new ArrayBuffer(4);
    new DataView(header).setInt32(0, this.channel, true);

    this.node.onaudioprocess = (e) => {
      if (this.ws.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      const down = downsample(input, this.ctx.sampleRate);
      const pcm = floatTo16BitPCM(down);
      const frame = new Uint8Array(4 + pcm.byteLength);
      frame.set(new Uint8Array(header), 0);
      frame.set(new Uint8Array(pcm.buffer), 4);
      this.ws.send(frame);
    };
    source.connect(this.node);
    this.node.connect(this.ctx.destination);
  }

  stop() {
    try { this.node && this.node.disconnect(); } catch (_) {}
    try { this.stream && this.stream.getTracks().forEach((t) => t.stop()); } catch (_) {}
    try { this.ctx && this.ctx.close(); } catch (_) {}
  }
}

// Один стереовход -> два канала конвейера: левый = участник A, правый = участник B.
// Нужно, когда оба микрофона воткнуты в ОДИН мини-USB адаптер: Windows показывает одно
// устройство, и обычный путь (два CaptureChannel по deviceId) посадил бы оба канала на него же,
// то есть записал бы одно и то же дважды. Здесь вместо двух устройств берётся одно, но
// расщепляется по стереоканалам.
//
// Обязательно ВЫКЛЮЧЕНА обработка (echoCancellation/noiseSuppression/autoGainControl): она
// сводит вход в моно, и стерео до нас просто не доедет — вместо двух участников получились бы
// две одинаковые дорожки. Для микрофона-на-участника эта обработка и не нужна: эхо динамиков
// здесь не пишется, а АРУ мешает кросс-канальному гейту сравнивать громкость каналов.
//
// Разведены ли микрофоны по L/R физически — вопрос к адаптеру, а не к коду: многие суммируют
// оба капсюля в один сигнал. Проверяется до записи: `py -3.11 -m app.test_mics`.
class StereoSplitCapture {
  constructor(deviceId, ws, channels = [0, 1]) {
    this.deviceId = deviceId;
    this.ws = ws;
    this.channels = channels;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: this.deviceId ? { exact: this.deviceId } : undefined,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 2,
      },
    });
    const track = this.stream.getAudioTracks()[0];
    // `channelCount: 2` в constraints — НЕточное ограничение: моно-грант это законный ответ
    // браузера, а не аномалия. Поэтому смотрим на факт. Прежнее `got === 1` пропускало моно
    // всюду, где адаптер не отдаёт channelCount вовсе (на Windows обычное дело): 0 !== 1, и
    // «неизвестно» проходило за «всё хорошо». Теперь неизвестное — повод проверить сигнал.
    const got = (track && track.getSettings && track.getSettings().channelCount) || 0;
    if (got === 1) {
      // Честнее упасть здесь, чем писать час и получить две одинаковые дорожки.
      this.stop();
      throw new Error("устройство отдало моно вместо стерео — расщеплять нечего");
    }

    this.ctx = new AudioContext();
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.node = this.ctx.createScriptProcessor(4096, 2, 2);
    const headers = this.channels.map((ch) => {
      const h = new ArrayBuffer(4);
      new DataView(h).setInt32(0, ch, true);
      return new Uint8Array(h);
    });
    source.connect(this.node);
    this.node.connect(this.ctx.destination);

    // Числом каналов буфера моно тут не поймать: createScriptProcessor(4096, 2, 2) апмиксит
    // моно-источник ДУБЛИРОВАНИЕМ, поэтому numberOfChannels всегда 2, а дорожки побайтово
    // одинаковы — протокол задваивается с одинаковыми тайм-кодами, а кросс-канальный гейт
    // сравнивает сигнал сам с собой. Единственный оставшийся признак — сам сигнал; смотрим на
    // него ДО того, как включим отправку, иначе задвоенные кадры уже уедут в WS.
    if (got !== 2 && !(await this._sourceLooksStereo())) {
      this.stop();
      throw new Error("устройство отдало моно вместо стерео — расщеплять нечего");
    }

    this.node.onaudioprocess = (e) => {
      if (this.ws.readyState !== WebSocket.OPEN) return;
      const n = Math.min(2, e.inputBuffer.numberOfChannels);
      for (let i = 0; i < n; i++) {
        const down = downsample(e.inputBuffer.getChannelData(i), this.ctx.sampleRate);
        const pcm = floatTo16BitPCM(down);
        const frame = new Uint8Array(4 + pcm.byteLength);
        frame.set(headers[i], 0);
        frame.set(new Uint8Array(pcm.buffer), 4);
        this.ws.send(frame);
      }
    };
  }

  /** Различаются ли L и R в первом же буфере с сигналом. Судим только по буферу, где сигнал
   *  ЕСТЬ: цифровая тишина в обоих каналах одинакова и у настоящего стерео, и по ней исправный
   *  адаптер был бы отвергнут. Если сигнала так и не появилось — не мешаем записи: сильную
   *  проверку (channelCount === 1) мы уже прошли, а глушить запись по догадке хуже. */
  _sourceLooksStereo(timeoutMs = 2000) {
    return new Promise((resolve) => {
      const done = (verdict) => {
        clearTimeout(timer);
        this.node.onaudioprocess = null;
        resolve(verdict);
      };
      const timer = setTimeout(() => done(true), timeoutMs);
      this.node.onaudioprocess = (e) => {
        const l = e.inputBuffer.getChannelData(0);
        const r = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : l;
        let differs = false, signal = false;
        for (let i = 0; i < l.length; i++) {
          if (l[i] !== 0 || r[i] !== 0) signal = true;
          if (l[i] !== r[i]) { differs = true; break; }
        }
        if (differs) done(true);
        else if (signal) done(false);
      };
    });
  }

  stop() {
    try { this.node && (this.node.onaudioprocess = null); } catch (_) {}
    try { this.node && this.node.disconnect(); } catch (_) {}
    try { this.stream && this.stream.getTracks().forEach((t) => t.stop()); } catch (_) {}
    try { this.ctx && this.ctx.close(); } catch (_) {}
  }
}

// Разовая запись с одного устройства в накопитель (для голосовых ответов анкеты).
class OneShotRecorder {
  constructor(deviceId) {
    this.deviceId = deviceId;
    this.chunks = [];
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: this.deviceId ? { exact: this.deviceId } : undefined,
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 1,
      },
    });
    this.ctx = new AudioContext();
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.node = this.ctx.createScriptProcessor(4096, 1, 1);
    this.node.onaudioprocess = (e) => {
      const down = downsample(e.inputBuffer.getChannelData(0), this.ctx.sampleRate);
      this.chunks.push(floatTo16BitPCM(down));
    };
    source.connect(this.node);
    this.node.connect(this.ctx.destination);
  }

  // Останавливает запись и возвращает единый Int16Array PCM 16кГц.
  stop() {
    try { this.node && this.node.disconnect(); } catch (_) {}
    try { this.stream && this.stream.getTracks().forEach((t) => t.stop()); } catch (_) {}
    try { this.ctx && this.ctx.close(); } catch (_) {}
    let total = 0;
    this.chunks.forEach((c) => (total += c.length));
    const out = new Int16Array(total);
    let off = 0;
    this.chunks.forEach((c) => { out.set(c, off); off += c.length; });
    return out;
  }
}

// Автозапись ответа анкеты по паузам, без единого клика (Блок 6). В отличие от
// `OneShotRecorder` (ручной старт/стоп кнопкой), сама решает, когда речь началась и
// закончилась — локальный VAD на клиенте, отдельный от серверного (Endpointer в
// backend/app/audio/endpointer.py). Это осознанный технический долг: два независимых VAD
// (JS и Python) могут разойтись по настройкам — задокументировано в README/плане, не
// устраняется в этой итерации. Построена на AudioWorklet (не ScriptProcessor) — новый код,
// нет причин тянуть устаревающий API.
//
// maxWaitForSpeechMs и maxUtteranceMs — намеренно разные вещи: первый — сколько ждать НАЧАЛА
// ответа (тишина совсем), второй — сколько максимум длится сам ответ, если он не прерывается
// паузой. Раньше в других формулировках эти два случая смешивались в одном таймауте.
class AutoRecorder {
  constructor(deviceId, opts = {}) {
    this.deviceId = deviceId;
    this.silenceToFinishMs = opts.silenceToFinishMs ?? 650;
    this.minSpeechMs = opts.minSpeechMs ?? 180;
    this.maxWaitForSpeechMs = opts.maxWaitForSpeechMs ?? 12000;
    this.maxUtteranceMs = opts.maxUtteranceMs ?? 60000;
    this.preRollMs = opts.preRollMs ?? 250;
    this.energyThreshold = opts.energyThreshold ?? 0.008;   // тот же порядок, что серверный VAD_ENERGY_THRESHOLD

    this._chunks = [];        // Int16Array-куски финальной записи (с момента открытия реплики)
    this._preRollBuf = [];    // кольцевой буфер до открытия реплики — попадает в начало записи
    this._preRollMaxSamples = Math.round((this.preRollMs * TARGET_RATE) / 1000);
    this._hadSpeech = false;
    this._speechRunMs = 0;
    this._silenceRunMs = 0;
    this._utteranceOpen = false;
    this._finished = false;
    this._onSilence = null;
    this._timeoutHandle = null;
  }

  /** cb(hadSpeech: bool) — вызывается один раз, когда запись остановлена (паузой или таймаутом). */
  onSilence(cb) { this._onSilence = cb; }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: this.deviceId ? { exact: this.deviceId } : undefined,
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 1,
      },
    });
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule("auto-recorder-worklet.js");
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "capture-processor");
    this.node.port.onmessage = (e) => this._onChunk(e.data);
    source.connect(this.node);

    this._scheduleTimeout(this.maxWaitForSpeechMs, () => this._finish());
  }

  _scheduleTimeout(ms, cb) {
    clearTimeout(this._timeoutHandle);
    this._timeoutHandle = setTimeout(cb, ms);
  }

  _rms(floatFrame) {
    let sum = 0;
    for (let i = 0; i < floatFrame.length; i++) sum += floatFrame[i] * floatFrame[i];
    return Math.sqrt(sum / (floatFrame.length || 1));
  }

  _trimPreRoll() {
    let total = 0;
    for (const c of this._preRollBuf) total += c.length;
    while (total > this._preRollMaxSamples && this._preRollBuf.length > 1) {
      total -= this._preRollBuf[0].length;
      this._preRollBuf.shift();
    }
  }

  _onChunk(floatChunk) {
    if (this._finished) return;
    const down = downsample(floatChunk, this.ctx.sampleRate);
    const pcm = floatTo16BitPCM(down);
    const chunkMs = (down.length / TARGET_RATE) * 1000;
    const isSpeech = this._rms(down) >= this.energyThreshold;

    if (!this._utteranceOpen) {
      this._preRollBuf.push(pcm);
      this._trimPreRoll();
      if (isSpeech) {
        this._speechRunMs += chunkMs;
        if (this._speechRunMs >= this.minSpeechMs) {
          this._utteranceOpen = true;
          this._hadSpeech = true;
          this._chunks = this._preRollBuf.slice();   // pre-roll входит в начало записи
          this._preRollBuf = [];
          this._silenceRunMs = 0;
          this._scheduleTimeout(this.maxUtteranceMs, () => this._finish());
        }
      } else {
        this._speechRunMs = 0;
      }
      return;
    }

    this._chunks.push(pcm);
    if (isSpeech) {
      this._silenceRunMs = 0;
    } else {
      this._silenceRunMs += chunkMs;
      // Хвостовая тишина до порога уже естественно попадает в запись — отдельный
      // "post-roll" не нужен, это то же самое явление.
      if (this._silenceRunMs >= this.silenceToFinishMs) this._finish();
    }
  }

  _finish() {
    if (this._finished) return;
    this._finished = true;
    clearTimeout(this._timeoutHandle);
    const hadSpeech = this._hadSpeech;
    this.stop();
    if (this._onSilence) this._onSilence(hadSpeech);
  }

  /** Останавливает захват. Возвращает Int16Array PCM 16кГц (пустой, если речи не было). */
  stop() {
    clearTimeout(this._timeoutHandle);
    try { if (this.node) this.node.port.onmessage = null; } catch (_) {}
    try { this.node && this.node.disconnect(); } catch (_) {}
    try { this.stream && this.stream.getTracks().forEach((t) => t.stop()); } catch (_) {}
    try { this.ctx && this.ctx.close(); } catch (_) {}
    let total = 0;
    this._chunks.forEach((c) => (total += c.length));
    const out = new Int16Array(total);
    let off = 0;
    this._chunks.forEach((c) => { out.set(c, off); off += c.length; });
    return out;
  }
}

window.AudioCapture = { CaptureChannel, StereoSplitCapture, OneShotRecorder, AutoRecorder };
