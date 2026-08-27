// AudioWorklet-процессор для автозаписи ответа анкеты (Блок 6) — работает на отдельном
// аудио-потоке, не завязан на основной JS event loop (в отличие от устаревающего
// ScriptProcessorNode, который используется для остального захвата в audio-capture.js —
// тот код стабилен и не трогаем, но новый код пишем сразу на AudioWorklet).
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = [];
    this._bufLen = 0;
    this._chunkSize = 4096;   // ~85-95мс при типичной частоте контекста 44.1/48кГц
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length) {
      this._buf.push(input[0].slice());
      this._bufLen += input[0].length;
      if (this._bufLen >= this._chunkSize) {
        const merged = new Float32Array(this._bufLen);
        let off = 0;
        for (const c of this._buf) { merged.set(c, off); off += c.length; }
        this.port.postMessage(merged, [merged.buffer]);
        this._buf = [];
        this._bufLen = 0;
      }
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
