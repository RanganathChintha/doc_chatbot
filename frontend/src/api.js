// Thin API client. Uses Vite's /api proxy → http://localhost:8000.

const BASE = '/api';

export async function listFiles(sessionId) {
  const res = await fetch(`${BASE}/files?session_id=${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error('Failed to load files');
  return res.json();
}

export async function uploadFiles(sessionId, files) {
  const form = new FormData();
  form.append('session_id', sessionId);
  for (const f of files) form.append('files', f);
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || 'Upload failed');
  }
  return res.json();
}

export async function ingestWiki(sessionId, wikiUrl, pat) {
  const res = await fetch(`${BASE}/wiki`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, wiki_url: wikiUrl, pat }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || 'Wiki indexing failed');
  }
  return res.json();
}

export async function clearFiles(sessionId) {
  const res = await fetch(`${BASE}/files?session_id=${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Clear failed');
  return res.json();
}

export async function deleteFile(sessionId, filename) {
  const res = await fetch(
    `${BASE}/file?session_id=${encodeURIComponent(sessionId)}&filename=${encodeURIComponent(filename)}`,
    { method: 'DELETE' },
  );
  if (!res.ok) throw new Error('Delete failed');
  return res.json();
}

export function thumbnailUrl(sessionId, filename) {
  return `${BASE}/thumbnail?session_id=${encodeURIComponent(sessionId)}&filename=${encodeURIComponent(filename)}`;
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
