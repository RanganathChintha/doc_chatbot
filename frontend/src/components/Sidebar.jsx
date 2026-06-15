import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { thumbnailUrl } from '../api.js';
import { ChatIcon, EditIcon, TrashIcon, UploadIcon, MoonIcon, SunIcon, CloseIcon, ChevronDownIcon } from './Icons.jsx';

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
  csv:  { bg: '#dcfce7', color: '#16a34a', label: 'CSV' },
  xlsx: { bg: '#dcfce7', color: '#16a34a', label: 'XLS' },
  xls:  { bg: '#dcfce7', color: '#16a34a', label: 'XLS' },
  pdf:  { bg: '#fee2e2', color: '#dc2626', label: 'PDF' },
};

const THUMB_EXTS = new Set(['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff']);
const WIKI_PREFIX = 'Azure Wiki - ';

const WIDTH_KEY = 'doc_chatbot.sidebarWidth';
const MIN_WIDTH = 220;
const MAX_WIDTH = 520;
const DEFAULT_WIDTH = 280;

const isUrl = (name) => /^https?:\/\//i.test(name);
const humanize = (seg) => seg.replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim();

function parseDocName(name) {
  if (name.startsWith(WIKI_PREFIX)) {
    const segs = name.slice(WIKI_PREFIX.length).split('/').filter(Boolean);
    const pathSegs = segs.slice(2);
    const leaf = pathSegs.length ? pathSegs[pathSegs.length - 1] : segs[segs.length - 1] || 'Page';
    const crumb = pathSegs.slice(0, -1).map(humanize).join(' › ');
    return { kind: 'wiki', title: humanize(leaf), sub: crumb || 'Azure DevOps Wiki' };
  }
  if (isUrl(name)) {
    let host = name;
    try { host = new URL(name).hostname; } catch { host = name.replace(/^https?:\/\//i, ''); }
    return { kind: 'url', title: name.replace(/^https?:\/\//i, ''), sub: host };
  }
  const ext = name.split('.').pop().toLowerCase();
  return { kind: ext, title: name, sub: `${(ext || 'file').toUpperCase()} document` };
}

const WikiIcon = () => (
  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
  </svg>
);

const GlobeIcon = () => (
  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <circle cx="12" cy="12" r="10" />
    <line x1="2" y1="12" x2="22" y2="12" />
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
  </svg>
);

function DocThumbnail({ sessionId, name, kind }) {
  const [failed, setFailed] = useState(false);

  if (kind === 'wiki') {
    return <span className="doc-thumb-tile wiki"><WikiIcon /></span>;
  }
  if (kind === 'url') {
    return <span className="doc-thumb-tile url"><GlobeIcon /></span>;
  }

  if (THUMB_EXTS.has(kind) && !failed) {
    return (
      <img
        className="doc-thumb"
        src={thumbnailUrl(sessionId, name)}
        alt=""
        onError={() => setFailed(true)}
      />
    );
  }

  const style = FILE_TYPE_STYLES[kind];
  if (style) {
    return (
      <span className="doc-thumb-badge" style={{ background: style.bg, color: style.color }}>
        {style.label}
      </span>
    );
  }
  return (
    <span className="doc-thumb-badge" style={{ background: '#f0eefc', color: '#4b40c5' }}>
      IMG
    </span>
  );
}

function DocSkeleton() {
  return (
    <div className="doc-skeleton doc-skeleton-pulse">
      <div className="doc-skeleton-thumb" />
      <div className="doc-skeleton-lines">
        <div className="doc-skeleton-line" />
        <div className="doc-skeleton-line" />
      </div>
    </div>
  );
}

export default function Sidebar({
  className = '',
  conversations,
  activeId,
  sessionId,
  indexedFiles,
  uploading,
  dark,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onRenameChat,
  onClearAll,
  onUpload,
  onIngestWiki,
  onClearFiles,
  onDeleteFile,
  onToggleDark,
  onClose,
}) {
  const [search, setSearch] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editingText, setEditingText] = useState('');
  const [wikiUrl, setWikiUrl] = useState('');
  const [wikiPat, setWikiPat] = useState('');
  const [docsOpen, setDocsOpen] = useState(true);
  const [docFilter, setDocFilter] = useState('');
  const fileRef = useRef(null);

  // ── Resizable width ──────────────────────────────────────────────
  const [width, setWidth] = useState(() => {
    const saved = parseInt(localStorage.getItem(WIDTH_KEY), 10);
    return saved >= MIN_WIDTH && saved <= MAX_WIDTH ? saved : DEFAULT_WIDTH;
  });

  useEffect(() => {
    try { localStorage.setItem(WIDTH_KEY, String(width)); } catch {}
  }, [width]);

  const onResizeMove = useCallback((e) => {
    setWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, e.clientX)));
  }, []);

  const stopResize = useCallback(() => {
    document.body.classList.remove('resizing-x');
    window.removeEventListener('mousemove', onResizeMove);
    window.removeEventListener('mouseup', stopResize);
  }, [onResizeMove]);

  const startResize = useCallback((e) => {
    e.preventDefault();
    document.body.classList.add('resizing-x');
    window.addEventListener('mousemove', onResizeMove);
    window.addEventListener('mouseup', stopResize);
  }, [onResizeMove, stopResize]);

  useEffect(() => stopResize, [stopResize]); // cleanup listeners on unmount

  const filteredDocs = useMemo(() => {
    const q = docFilter.trim().toLowerCase();
    if (!q) return indexedFiles;
    return indexedFiles.filter((name) => {
      const info = parseDocName(name);
      return (
        name.toLowerCase().includes(q) ||
        info.title.toLowerCase().includes(q) ||
        info.sub.toLowerCase().includes(q)
      );
    });
  }, [indexedFiles, docFilter]);

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

  const onWikiSubmit = async (e) => {
    e.preventDefault();
    const url = wikiUrl.trim();
    const pat = wikiPat.trim();
    if (!url || uploading) return;
    await onIngestWiki({ wikiUrl: url, pat: pat || undefined });
    setWikiUrl('');
    setWikiPat('');
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

  const onConvKeyDown = (e, c) => {
    if (e.target !== e.currentTarget) return; // ignore keys from inner buttons/input
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onSelectChat(c.id);
    }
  };

  return (
    <aside className={`sidebar ${className}`} style={{ '--sidebar-width': `${width}px` }}>
      <div className="sidebar-header">
        <div className="brand">DocPilot</div>
        <div className="sidebar-header-actions">
          <button
            className="header-icon-btn"
            onClick={onToggleDark}
            aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
            title={dark ? 'Light mode' : 'Dark mode'}
          >
            {dark ? <SunIcon /> : <MoonIcon />}
          </button>
          <button className="sidebar-close" onClick={onClose} aria-label="Close menu">
            <CloseIcon />
          </button>
        </div>
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
                  role="button"
                  tabIndex={0}
                  className={`conv-item ${c.id === activeId ? 'active' : ''}`}
                  onClick={() => onSelectChat(c.id)}
                  onKeyDown={(e) => onConvKeyDown(e, c)}
                  aria-label={c.title}
                  aria-current={c.id === activeId ? 'true' : undefined}
                >
                  <span className="conv-icon"><ChatIcon /></span>
                  {editingId === c.id ? (
                    <input
                      autoFocus
                      className="conv-edit"
                      value={editingText}
                      onChange={(e) => setEditingText(e.target.value)}
                      onBlur={() => commitEdit(c.id)}
                      onKeyDown={(e) => {
                        e.stopPropagation();
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
                      aria-label="Rename"
                      onClick={(e) => { e.stopPropagation(); startEdit(c); }}
                    >
                      <EditIcon />
                    </button>
                    <button
                      className="icon-btn"
                      aria-label="Delete conversation"
                      onClick={(e) => { e.stopPropagation(); onDeleteChat(c.id); }}
                    >
                      <TrashIcon />
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
          <button
            className="docs-toggle"
            onClick={() => setDocsOpen((v) => !v)}
            aria-expanded={docsOpen}
          >
            <span className={`docs-chevron ${docsOpen ? 'open' : ''}`}><ChevronDownIcon /></span>
            <span>Documents</span>
            {indexedFiles.length > 0 && <span className="docs-count">{indexedFiles.length}</span>}
          </button>
          {indexedFiles.length > 0 && (
            <button className="link-btn" onClick={onClearFiles}>Clear all</button>
          )}
        </div>
        <button
          className="upload-btn"
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
        >
          <UploadIcon /> {uploading ? 'Indexing…' : 'Upload files'}
        </button>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept=".pdf,.png,.jpg,.jpeg,.bmp,.tiff,.csv,.xlsx,.xls"
          style={{ display: 'none' }}
          onChange={onFilePick}
        />
        <form className="wiki-form" onSubmit={onWikiSubmit}>
          <input
            className="wiki-input"
            placeholder="Web URL (parent page)"
            value={wikiUrl}
            onChange={(e) => setWikiUrl(e.target.value)}
            disabled={uploading}
          />
          <input
            className="wiki-input"
            type="password"
            placeholder="Auth token (optional)"
            value={wikiPat}
            onChange={(e) => setWikiPat(e.target.value)}
            disabled={uploading}
          />
          <button
            className="wiki-submit"
            type="submit"
            disabled={uploading || !wikiUrl.trim()}
          >
            {uploading ? 'Indexing...' : 'Fetch pages'}
          </button>
        </form>
        {indexedFiles.length > 6 && docsOpen && (
          <input
            className="docs-filter"
            placeholder="Filter documents…"
            value={docFilter}
            onChange={(e) => setDocFilter(e.target.value)}
          />
        )}
        {docsOpen && (
          <div className="docs-list">
            {uploading && <DocSkeleton />}
            {filteredDocs.map((name) => {
              const info = parseDocName(name);
              return (
                <div className="doc-item" key={name}>
                  <DocThumbnail sessionId={sessionId} name={name} kind={info.kind} />
                  <div className="doc-meta">
                    <span className="doc-title" title={name}>{info.title}</span>
                    <span className="doc-sub" title={info.sub}>{info.sub}</span>
                  </div>
                  <button
                    className="doc-del"
                    aria-label={`Remove ${name}`}
                    onClick={() => onDeleteFile(name)}
                  >
                    &times;
                  </button>
                </div>
              );
            })}
            {indexedFiles.length === 0 && !uploading && (
              <div className="docs-empty">No documents indexed yet.</div>
            )}
            {indexedFiles.length > 0 && filteredDocs.length === 0 && (
              <div className="docs-empty">No documents match “{docFilter}”.</div>
            )}
          </div>
        )}
      </div>

      <div
        className="sidebar-resizer"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize sidebar"
        title="Drag to resize · double-click to reset"
        onMouseDown={startResize}
        onDoubleClick={() => setWidth(DEFAULT_WIDTH)}
      />
    </aside>
  );
}
