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

window.AudioCapture = { CaptureChannel, OneShotRecorder };
