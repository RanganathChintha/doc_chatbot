import { useMemo, useRef, useState } from 'react';
import { thumbnailUrl } from '../api.js';

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

const FILE_TYPE_STYLES = {
  pdf:  { bg: '#fee2e2', color: '#dc2626', label: 'PDF' },
  csv:  { bg: '#dcfce7', color: '#16a34a', label: 'CSV' },
  xlsx: { bg: '#dcfce7', color: '#16a34a', label: 'XLS' },
  xls:  { bg: '#dcfce7', color: '#16a34a', label: 'XLS' },
};

const IMAGE_EXTS = new Set(['png', 'jpg', 'jpeg', 'bmp', 'tiff']);
const THUMB_EXTS = new Set(['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff']);

function DocThumbnail({ sessionId, name }) {
  const ext = name.split('.').pop().toLowerCase();
  const [failed, setFailed] = useState(false);

  if (THUMB_EXTS.has(ext) && !failed) {
    return (
      <img
        className="doc-thumb"
        src={thumbnailUrl(sessionId, name)}
        alt=""
        onError={() => setFailed(true)}
      />
    );
  }

  const style = FILE_TYPE_STYLES[ext];
  if (style) {
    return (
      <span className="doc-thumb-badge" style={{ background: style.bg, color: style.color }}>
        {style.label}
      </span>
    );
  }
  // generic image fallback
  return (
    <span className="doc-thumb-badge" style={{ background: '#f0eefc', color: '#4b40c5' }}>
      IMG
    </span>
  );
}

export default function Sidebar({
  conversations,
  activeId,
  sessionId,
  indexedFiles,
  uploading,
  crawling,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onRenameChat,
  onClearAll,
  onUpload,
  onCrawl,
  onClearFiles,
  onDeleteFile,
}) {
  const [search, setSearch] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editingText, setEditingText] = useState('');
  const [crawlUrl, setCrawlUrl] = useState('');
  const [renderJavascript, setRenderJavascript] = useState(true);
  const [fullSite, setFullSite] = useState(true);
  const [maxDepth, setMaxDepth] = useState(1);
  const [maxPages, setMaxPages] = useState(5);
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

  const submitCrawl = (e) => {
    e.preventDefault();
    const urls = crawlUrl
      .split(/\s+/)
      .map((url) => url.trim())
      .filter(Boolean);
    if (urls.length === 0) return;
    onCrawl({
      urls,
      fullSite,
      maxDepth,
      maxPages,
      renderJavascript,
      renderTimeout: 30,
    });
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
                      onClick={(e) => { e.stopPropagation(); startEdit(c); }}
                    >
                      ✎
                    </button>
                    <button
                      className="icon-btn"
                      title="Delete"
                      onClick={(e) => { e.stopPropagation(); onDeleteChat(c.id); }}
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
            <button className="link-btn" onClick={onClearFiles}>Clear all</button>
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
        <form className="crawl-form" onSubmit={submitCrawl}>
          <textarea
            className="crawl-input"
            rows={2}
            placeholder="Paste internal URL"
            value={crawlUrl}
            onChange={(e) => setCrawlUrl(e.target.value)}
            disabled={crawling}
          />
          <div className="crawl-controls">
            <label className="crawl-toggle">
              <input
                type="checkbox"
                checked={renderJavascript}
                onChange={(e) => setRenderJavascript(e.target.checked)}
                disabled={crawling}
              />
              Render JS
            </label>
            <label className="crawl-toggle">
              <input
                type="checkbox"
                checked={fullSite}
                onChange={(e) => setFullSite(e.target.checked)}
                disabled={crawling}
              />
              Full site
            </label>
            <label className="crawl-number">
              Depth
              <input
                type="number"
                min="0"
                max="5"
                value={maxDepth}
                onChange={(e) => setMaxDepth(Number(e.target.value))}
                disabled={crawling || fullSite}
              />
            </label>
            <label className="crawl-number">
              Pages
              <input
                type="number"
                min="1"
                max="50"
                value={maxPages}
                onChange={(e) => setMaxPages(Number(e.target.value))}
                disabled={crawling || fullSite}
              />
            </label>
          </div>
          <button className="crawl-btn" type="submit" disabled={crawling || !crawlUrl.trim()}>
            {crawling ? 'Crawling...' : 'Crawl URL'}
          </button>
        </form>
        <div className="docs-list">
          {indexedFiles.map((name) => (
            <div className="doc-item" key={name} title={name}>
              <DocThumbnail sessionId={sessionId} name={name} />
              <span className="doc-name">{name}</span>
              <button
                className="doc-del"
                title="Remove file"
                onClick={() => onDeleteFile(name)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
