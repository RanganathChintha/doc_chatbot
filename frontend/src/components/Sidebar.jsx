import { useMemo, useRef, useState } from 'react';

function groupByTime(convs) {
  const now = Date.now();
  const day = 24 * 60 * 60 * 1000;
  const groups = { Today: [], Yesterday: [], 'Last 7 Days': [], 'Last 30 Days': [], Older: [] };
  for (const c of convs) {
    const age = now - c.updatedAt;
    if (age < day) groups.Today.push(c);
    else if (age < 2 * day) groups.Yesterday.push(c);
    else if (age < 7 * day) groups['Last 7 Days'].push(c);
    else if (age < 30 * day) groups['Last 30 Days'].push(c);
    else groups.Older.push(c);
  }
  return groups;
}

export default function Sidebar({
  conversations,
  activeId,
  indexedFiles,
  uploading,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onRenameChat,
  onClearAll,
  onUpload,
  onClearFiles,
}) {
  const [search, setSearch] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editingText, setEditingText] = useState('');
  const fileRef = useRef(null);

  const filtered = useMemo(() => {
    if (!search.trim()) return conversations;
    const s = search.toLowerCase();
    return conversations.filter(
      (c) => c.title.toLowerCase().includes(s) || c.messages.some((m) => m.content.toLowerCase().includes(s)),
    );
  }, [conversations, search]);

  const groups = useMemo(() => groupByTime(filtered), [filtered]);

  const onFilePick = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) onUpload(files);
    e.target.value = '';
  };

  const startEdit = (c) => {
    setEditingId(c.id);
    setEditingText(c.title);
  };

  const commitEdit = (id) => {
    onRenameChat(id, editingText);
    setEditingId(null);
    setEditingText('');
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand">DocPilot</div>
      </div>

      <button className="new-chat-btn" onClick={onNewChat}>
        <span className="plus">+</span> New chat
      </button>

      <div className="search-row">
        <input
          className="search-input"
          placeholder="Search conversations…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="conv-section-head">
        <span>Your conversations</span>
        <button className="link-btn" onClick={onClearAll}>Clear All</button>
      </div>

      <div className="conv-list">
        {Object.entries(groups).map(([label, items]) =>
          items.length === 0 ? null : (
            <div key={label} className="conv-group">
              {label !== 'Today' && <div className="group-label">{label}</div>}
              {items.map((c) => (
                <div
                  key={c.id}
                  className={`conv-item ${c.id === activeId ? 'active' : ''}`}
                  onClick={() => onSelectChat(c.id)}
                >
                  <span className="conv-icon" aria-hidden>💬</span>
                  {editingId === c.id ? (
                    <input
                      autoFocus
                      className="conv-edit"
                      value={editingText}
                      onChange={(e) => setEditingText(e.target.value)}
                      onBlur={() => commitEdit(c.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') commitEdit(c.id);
                        if (e.key === 'Escape') {
                          setEditingId(null);
                          setEditingText('');
                        }
                      }}
                    />
                  ) : (
                    <span className="conv-title" title={c.title}>{c.title}</span>
                  )}
                  <span className="conv-actions">
                    <button
                      className="icon-btn"
                      title="Rename"
                      onClick={(e) => {
                        e.stopPropagation();
                        startEdit(c);
                      }}
                    >
                      ✎
                    </button>
                    <button
                      className="icon-btn"
                      title="Delete"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteChat(c.id);
                      }}
                    >
                      🗑
                    </button>
                  </span>
                </div>
              ))}
            </div>
          ),
        )}
        {filtered.length === 0 && <div className="empty-state">No conversations match.</div>}
      </div>

      <div className="docs-section">
        <div className="docs-head">
          <span>Documents</span>
          {indexedFiles.length > 0 && (
            <button className="link-btn" onClick={onClearFiles}>Clear</button>
          )}
        </div>
        <button
          className="upload-btn"
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? 'Indexing…' : '⬆ Upload files'}
        </button>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept=".pdf,.png,.jpg,.jpeg,.bmp,.tiff,.csv,.xlsx,.xls"
          style={{ display: 'none' }}
          onChange={onFilePick}
        />
        <div className="docs-list">
          {indexedFiles.map((name) => (
            <div className="doc-item" key={name} title={name}>📄 {name}</div>
          ))}
        </div>
      </div>
    </aside>
  );
}
