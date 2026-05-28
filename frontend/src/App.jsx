import { useCallback, useEffect, useRef, useState } from 'react';
import Sidebar from './components/Sidebar.jsx';
import ChatPane from './components/ChatPane.jsx';
import InputBox from './components/InputBox.jsx';
import { chatStream, listFiles, resetSession, uploadFiles, clearFiles } from './api.js';

const STORAGE_KEY = 'doc_chatbot.conversations.v1';

function newConversation() {
  return {
    id: crypto.randomUUID(),
    title: 'New chat',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [], // {role, content, sources?}
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

function saveConversations(list) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}

export default function App() {
  const [conversations, setConversations] = useState(() => loadConversations() ?? [newConversation()]);
  const [activeId, setActiveId] = useState(() => {
    const stored = loadConversations();
    return (stored && stored[0].id) || conversations[0].id;
  });
  const [indexedFiles, setIndexedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  const active = conversations.find((c) => c.id === activeId) ?? conversations[0];

  useEffect(() => {
    saveConversations(conversations);
  }, [conversations]);

  useEffect(() => {
    listFiles().then((d) => setIndexedFiles(d.files || [])).catch(() => {});
  }, []);

  const updateConversation = useCallback((id, patch) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch, updatedAt: Date.now() } : c)));
  }, []);

  const onNewChat = () => {
    const c = newConversation();
    setConversations((prev) => [c, ...prev]);
    setActiveId(c.id);
  };

  const onSelectChat = (id) => setActiveId(id);

  const onDeleteChat = (id) => {
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
    resetSession(id).catch(() => {});
  };

  const onClearAll = () => {
    if (!confirm('Delete all conversations?')) return;
    const fresh = newConversation();
    setConversations([fresh]);
    setActiveId(fresh.id);
  };

  const onRenameChat = (id, title) => updateConversation(id, { title: title.trim() || 'New chat' });

  const onUpload = async (files) => {
    setUploading(true);
    setError(null);
    try {
      const res = await uploadFiles(files);
      setIndexedFiles(res.indexed_files || []);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setUploading(false);
    }
  };

  const onClearFiles = async () => {
    if (!confirm('Remove all uploaded documents from the index?')) return;
    try {
      await clearFiles();
      setIndexedFiles([]);
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  const onSend = async (text) => {
    const trimmed = text.trim();
    if (!trimmed || streaming) return;
    if (indexedFiles.length === 0) {
      setError('Upload at least one document before chatting.');
      return;
    }

    setError(null);
    const userMsg = { role: 'user', content: trimmed };
    const assistantMsg = { role: 'assistant', content: '', sources: [] };

    // Snapshot active conversation; auto-title from first user message.
    const nextMessages = [...active.messages, userMsg, assistantMsg];
    const patch = { messages: nextMessages };
    if (active.messages.length === 0) patch.title = trimmed.slice(0, 40);
    updateConversation(active.id, patch);

    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      for await (const event of chatStream(active.id, trimmed, { signal: controller.signal })) {
        if (event.type === 'token') {
          assistantMsg.content += event.text;
          updateConversation(active.id, { messages: [...nextMessages.slice(0, -1), { ...assistantMsg }] });
        } else if (event.type === 'sources') {
          assistantMsg.sources = event.sources;
          updateConversation(active.id, { messages: [...nextMessages.slice(0, -1), { ...assistantMsg }] });
        } else if (event.type === 'error') {
          assistantMsg.content = `⚠️ ${event.message}`;
          updateConversation(active.id, { messages: [...nextMessages.slice(0, -1), { ...assistantMsg }] });
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        assistantMsg.content = assistantMsg.content || `⚠️ ${e.message || e}`;
        updateConversation(active.id, { messages: [...nextMessages.slice(0, -1), { ...assistantMsg }] });
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  };

  const onStop = () => abortRef.current?.abort();

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        indexedFiles={indexedFiles}
        uploading={uploading}
        onNewChat={onNewChat}
        onSelectChat={onSelectChat}
        onDeleteChat={onDeleteChat}
        onRenameChat={onRenameChat}
        onClearAll={onClearAll}
        onUpload={onUpload}
        onClearFiles={onClearFiles}
      />
      <main className="main-pane">
        <div className="main-inner">
          <ChatPane conversation={active} />
          {error && <div className="error-banner">{error}</div>}
          <InputBox
            disabled={uploading}
            streaming={streaming}
            onSend={onSend}
            onStop={onStop}
            placeholder={
              indexedFiles.length === 0
                ? 'Upload a document on the left to start chatting…'
                : "What's in your mind?"
            }
          />
        </div>
      </main>
    </div>
  );
}
