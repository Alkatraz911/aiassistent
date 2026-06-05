// Минимальный preload: пробрасываем адрес backend в renderer.
const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("APP", {
  backendHttp: process.env.BACKEND_HTTP || "http://127.0.0.1:8000",
  backendWs: process.env.BACKEND_WS || "ws://127.0.0.1:8000",
});
