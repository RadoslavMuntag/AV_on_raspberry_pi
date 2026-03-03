const stateEl = document.getElementById('state');
const clientIdEl = document.getElementById('clientId');
const feedSel = document.getElementById('feedSel');
let ws;
let wsSeq = 0;
let currentMode = 'idle';

function clientId() { return clientIdEl.value || 'web-local'; }

async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }
  return res.json();
}

async function acquire() {
  await post('/api/controller/acquire', {client_id: clientId()});
}

async function releaseCtrl() {
  await post('/api/controller/release', {client_id: clientId()});
}

async function setMode() {
  const mode = document.getElementById('modeSel').value;
  await post('/api/mode', { mode });
  currentMode = mode; // optimistic update
}

async function drive() {
  const left = Number(document.getElementById('left').value);
  const right = Number(document.getElementById('right').value);
  await fetch(`/api/control/drive?client_id=${encodeURIComponent(clientId())}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({left, right})
  });
}

async function stopNow() {
  document.getElementById('left').value = 0;
  document.getElementById('right').value = 0;
  await drive();
  await setModeSafeStop();
}

async function setModeSafeStop() {
  await post('/api/mode', {mode: 'safe_stop'});
}

async function heartbeat() {
  if (currentMode !== 'manual') return;
  try {
    await post('/api/controller/heartbeat', {client_id: clientId()});
  } catch (_) {}
}

function wsBaseUrl() {
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`;
}

function currentWsPath() {
  const frame = feedSel?.value || 'telemetry';
  if (frame === 'telemetry') return '/ws/telemetry';
  return `/ws/pipeline?frame=${encodeURIComponent(frame)}`;
}

function connectWs() {
  wsSeq += 1;
  const seq = wsSeq;

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.onclose = null; // avoid reconnect from intentional close
    ws.close();
  }

  ws = new WebSocket(`${wsBaseUrl()}${currentWsPath()}`);

  ws.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      stateEl.textContent = JSON.stringify(payload, null, 2);
    } catch {
      stateEl.textContent = event.data;
    }
  };

  ws.onclose = () => {
    setTimeout(() => {
      if (seq === wsSeq) connectWs();
    }, 1000);
  };
}

if (feedSel) {
  feedSel.addEventListener('change', connectWs);
}

connectWs();
setInterval(heartbeat, 500);