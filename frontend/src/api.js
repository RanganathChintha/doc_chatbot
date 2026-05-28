// Thin API client. Uses Vite's /api proxy → http://localhost:8000.

const BASE = '/api';

export async function listFiles() {
  const res = await fetch(`${BASE}/files`);
  if (!res.ok) throw new Error('Failed to load files');
  return res.json();
}

export async function uploadFiles(files) {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || 'Upload failed');
  }
  return res.json();
}

export async function clearFiles() {
  const res = await fetch(`${BASE}/files`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Clear failed');
  return res.json();
}

export async function resetSession(sessionId) {
  const res = await fetch(`${BASE}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error('Reset failed');
  return res.json();
}

// Streaming chat: yields {type, ...} events as they arrive from SSE.
export async function* chatStream(sessionId, message, { signal } = {}) {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  });
  if (!res.ok || !res.body) {
    const t = await res.text().catch(() => '');
    throw new Error(t || `Chat failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by blank line.
    let idx;
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      for (const line of raw.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const json = line.slice(5).trim();
        if (!json) continue;
        try {
          yield JSON.parse(json);
        } catch {
          // ignore malformed frame
        }
      }
    }
  }
}
