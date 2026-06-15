import { useCallback, useEffect, useRef, useState } from 'react';
import Sidebar from './components/Sidebar.jsx';
import ChatPane from './components/ChatPane.jsx';
import InputBox from './components/InputBox.jsx';
import Toast from './components/Toast.jsx';
import ConfirmDialog from './components/ConfirmDialog.jsx';
import { MenuIcon, UploadIcon } from './components/Icons.jsx';
import { chatStream, listFiles, resetSession, uploadFiles, clearFiles, deleteFile, ingestWiki } from './api.js';

const STORAGE_KEY = 'doc_chatbot.conversations.v1';
const DARK_KEY = 'doc_chatbot.dark';

function newConversation() {
  return {
    id: crypto.randomUUID(),
    title: 'New chat',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
    files: [],
  };
}

function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) return null;
    return parsed;
  } catch {
    return null;
  }
}

// Persist conversations, recovering from quota errors by trimming history.
function persistConversations(list) {
  const attempt = (data) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  };
  try {
    attempt(list);
    return { ok: true };
  } catch {
    // Trim each conversation to its most recent messages, then retry.
    try {
      const trimmed = list.map((c) => ({ ...c, messages: c.messages.slice(-40) }));
      attempt(trimmed);
      return { ok: true, trimmed: true };
    } catch {
      return { ok: false };
    }
  }
}

export default function App() {
  const [conversations, setConversations] = useState(() => loadConversations() ?? [newConversation()]);
  const [activeId, setActiveId] = useState(() => {
    const stored = loadConversations();
    return (stored && stored[0].id) || conversations[0].id;
  });
  const [uploading, setUploading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [toasts, setToasts] = useState([]);
  const [confirmState, setConfirmState] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [dark, setDark] = useState(() => {
    try { return localStorage.getItem(DARK_KEY) === 'true'; } catch { return false; }
  });
  const abortRef = useRef(null);
  const dragDepth = useRef(0);
  const quotaWarned = useRef(false);

  const active = conversations.find((c) => c.id === activeId) ?? conversations[0];
  const indexedFiles = active?.files ?? [];
  const hasDocs = indexedFiles.length > 0;

  // ── Toasts ────────────────────────────────────────────────────────────────
  const showToast = useCallback((message, type = 'error') => {
    const id = crypto.randomUUID();
    setToasts((prev) => [...prev, { id, message, type }]);
  }, []);
  const dismissToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // ── Confirm modal (promise-based) ───────────────────────────────────────────
  const askConfirm = useCallback((opts) => {
    return new Promise((resolve) => {
      setConfirmState({ ...opts, resolve });
    });
  }, []);
  const resolveConfirm = (value) => {
    setConfirmState((s) => {
      s?.resolve(value);
      return null;
    });
  };

  // ── Persistence ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const res = persistConversations(conversations);
    if (!res.ok && !quotaWarned.current) {
      quotaWarned.current = true;
      showToast('Storage is full — older history may not be saved.', 'error');
    } else if (res.ok) {
      quotaWarned.current = false;
    }
  }, [conversations, showToast]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    try { localStorage.setItem(DARK_KEY, dark ? 'true' : ''); } catch {}
  }, [dark]);

  const updateConversation = useCallback((id, patch) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch, updatedAt: Date.now() } : c)));
  }, []);

  const setConversationFiles = useCallback((id, files) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, files } : c)));
  }, []);

  useEffect(() => {
    if (!activeId) return;
    listFiles(activeId)
      .then((d) => setConversationFiles(activeId, d.files || []))
      .catch(() => {});
  }, [activeId, setConversationFiles]);

  const onNewChat = () => {
    const c = newConversation();
    setConversations((prev) => [c, ...prev]);
    setActiveId(c.id);
    setSidebarOpen(false);
  };

  const onSelectChat = (id) => {
    setActiveId(id);
    setSidebarOpen(false);
  };

  const onDeleteChat = async (id) => {
    const ok = await askConfirm({
      title: 'Delete conversation',
      message: 'This conversation and its indexed documents will be removed. Continue?',
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (next.length === 0) {
        const fresh = newConversation();
        setActiveId(fresh.id);
        return [fresh];
      }
      if (id === activeId) setActiveId(next[0].id);
      return next;
    });
    clearFiles(id).catch(() => {});
    resetSession(id).catch(() => {});
  };

  const onClearAll = async () => {
    const ok = await askConfirm({
      title: 'Delete all conversations',
      message: 'Every conversation will be permanently deleted. This cannot be undone.',
      confirmLabel: 'Delete all',
      danger: true,
    });
    if (!ok) return;
    const fresh = newConversation();
    setConversations([fresh]);
    setActiveId(fresh.id);
  };

  const onRenameChat = (id, title) => updateConversation(id, { title: title.trim() || 'New chat' });

  const onUpload = async (files) => {
    const chatId = active.id;
    setUploading(true);
    try {
      const res = await uploadFiles(chatId, files);
      setConversationFiles(chatId, res.indexed_files || []);
      const count = res.new_chunks ?? 0;
      showToast(`Indexed ${count} chunk${count !== 1 ? 's' : ''} from ${files.length} file${files.length !== 1 ? 's' : ''}`, 'success');
    } catch (e) {
      showToast(String(e.message || e));
    } finally {
      setUploading(false);
    }
  };

  const onIngestWiki = async ({ wikiUrl, pat }) => {
    const chatId = active.id;
    setUploading(true);
    try {
      const res = await ingestWiki(chatId, wikiUrl, pat);
      setConversationFiles(chatId, res.indexed_files || []);
      const count = res.new_chunks ?? 0;
      showToast(`Indexed ${count} chunk${count !== 1 ? 's' : ''} from wiki`, 'success');
    } catch (e) {
      showToast(String(e.message || e));
    } finally {
      setUploading(false);
    }
  };

  const onClearFiles = async () => {
    const ok = await askConfirm({
      title: 'Remove documents',
      message: 'Remove all uploaded documents from this chat?',
      confirmLabel: 'Remove all',
      danger: true,
    });
    if (!ok) return;
    const chatId = active.id;
    try {
      await clearFiles(chatId);
      setConversationFiles(chatId, []);
      showToast('All documents removed', 'success');
    } catch (e) {
      showToast(String(e.message || e));
    }
  };

  const onDeleteFile = async (filename) => {
    const ok = await askConfirm({
      title: 'Remove document',
      message: `Remove "${filename}" from this chat?`,
      confirmLabel: 'Remove',
      danger: true,
    });
    if (!ok) return;
    const chatId = active.id;
    try {
      const res = await deleteFile(chatId, filename);
      setConversationFiles(chatId, res.indexed_files || []);
      showToast('File removed', 'success');
    } catch (e) {
      showToast(String(e.message || e));
    }
  };

  // Stream an assistant reply onto `baseMessages` (which must already end with
  // the triggering user message). Shared by send and regenerate.
  const streamInto = useCallback(async (chatId, baseMessages, userText) => {
    const assistantMsg = { id: crypto.randomUUID(), role: 'assistant', content: '', sources: [] };
    const nextMessages = [...baseMessages, assistantMsg];
    updateConversation(chatId, { messages: nextMessages });

    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;

    const flush = () => updateConversation(chatId, { messages: [...nextMessages.slice(0, -1), { ...assistantMsg }] });

    try {
      for await (const event of chatStream(chatId, userText, { signal: controller.signal })) {
        if (event.type === 'token') {
          assistantMsg.content += event.text;
          flush();
        } else if (event.type === 'sources') {
          assistantMsg.sources = event.sources;
          flush();
        } else if (event.type === 'error') {
          assistantMsg.content = `⚠️ ${event.message}`;
          flush();
        }
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        assistantMsg.stopped = true;
      } else {
        assistantMsg.content = assistantMsg.content || `⚠️ ${e.message || e}`;
      }
      flush();
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }, [updateConversation]);

  const onSend = async (text) => {
    const trimmed = text.trim();
    if (!trimmed || streaming) return;

    const userMsg = { id: crypto.randomUUID(), role: 'user', content: trimmed };
    const base = [...active.messages, userMsg];
    if (active.messages.length === 0) updateConversation(active.id, { title: trimmed.slice(0, 40) });

    await streamInto(active.id, base, trimmed);
  };

  const onRegenerate = async () => {
    if (streaming) return;
    const msgs = active.messages;
    let lastUserIdx = -1;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === 'user') { lastUserIdx = i; break; }
    }
    if (lastUserIdx === -1) return;
    const userText = msgs[lastUserIdx].content;
    const base = msgs.slice(0, lastUserIdx + 1);
    await streamInto(active.id, base, userText);
  };

  const onStop = () => abortRef.current?.abort();

  // ── Drag & drop upload ──────────────────────────────────────────────────────
  const hasFiles = (e) => Array.from(e.dataTransfer?.types || []).includes('Files');

  const onDragEnter = (e) => {
    if (!hasFiles(e) || uploading) return;
    e.preventDefault();
    dragDepth.current += 1;
    setDragging(true);
  };
  const onDragOver = (e) => {
    if (!hasFiles(e) || uploading) return;
    e.preventDefault();
  };
  const onDragLeave = (e) => {
    if (!hasFiles(e)) return;
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragging(false);
  };
  const onDrop = (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    if (uploading) return;
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length > 0) onUpload(files);
  };

  // ── Keyboard shortcuts ──────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        onNewChat();
      } else if (e.key === 'Escape' && streaming) {
        onStop();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streaming]);

  const disabledReason = uploading
    ? 'Indexing documents…'
    : !hasDocs
      ? 'Upload a document to start chatting'
      : '';

  return (
    <>
      <div
        className={`app ${dragging ? 'dragging' : ''}`}
        onDragEnter={onDragEnter}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        {sidebarOpen && <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />}
        <Sidebar
          className={sidebarOpen ? 'open' : ''}
          conversations={conversations}
          activeId={activeId}
          sessionId={active?.id}
          indexedFiles={indexedFiles}
          uploading={uploading}
          dark={dark}
          onNewChat={onNewChat}
          onSelectChat={onSelectChat}
          onDeleteChat={onDeleteChat}
          onRenameChat={onRenameChat}
          onClearAll={onClearAll}
          onUpload={onUpload}
          onIngestWiki={onIngestWiki}
          onClearFiles={onClearFiles}
          onDeleteFile={onDeleteFile}
          onToggleDark={() => setDark((d) => !d)}
          onClose={() => setSidebarOpen(false)}
        />
        <main className="main-pane">
          <div className="mobile-topbar">
            <button className="menu-btn" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
              <MenuIcon />
            </button>
            <span className="mobile-brand">DocPilot</span>
          </div>
          <div className="main-inner">
            <ChatPane
              conversation={active}
              streaming={streaming}
              hasDocs={hasDocs}
              onSend={onSend}
              onRegenerate={onRegenerate}
            />
            <InputBox
              disabled={uploading || !hasDocs}
              disabledReason={disabledReason}
              streaming={streaming}
              onSend={onSend}
              onStop={onStop}
              placeholder={hasDocs ? "What's on your mind?" : 'Upload a document on the left to start chatting…'}
            />
          </div>
        </main>

        {dragging && (
          <div className="drop-overlay">
            <div className="drop-card">
              <UploadIcon />
              <span>Drop files to index them</span>
            </div>
          </div>
        )}
      </div>

      <div className="toast-container">
        {toasts.map((t) => (
          <Toast key={t.id} message={t.message} type={t.type} onClose={() => dismissToast(t.id)} />
        ))}
      </div>

      <ConfirmDialog
        open={!!confirmState}
        title={confirmState?.title}
        message={confirmState?.message}
        confirmLabel={confirmState?.confirmLabel}
        danger={confirmState?.danger}
        onConfirm={() => resolveConfirm(true)}
        onCancel={() => resolveConfirm(false)}
      />
    </>
  );
}
