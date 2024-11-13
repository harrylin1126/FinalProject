
export var ws


export const initializeWebSocket = () => {
  if (!ws || ws.readyState === WebSocket.CLOSED) {
    ws = new WebSocket("ws://localhost:9999/socket");

    ws.onopen = function (e) {
      console.log("WebSocket connected");
      ws.send(JSON.stringify({flag:"aaa"}))
    };

    ws.onclose = function (e) {
      console.log("WebSocket closed");
      ws.close(1000, "success");
    };
  }
};