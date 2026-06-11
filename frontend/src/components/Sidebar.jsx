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
  csv:  { bg: '#dcfce7', color: '#16a34a', label: 'CSV' },
  xlsx: { bg: '#dcfce7', color: '#16a34a', label: 'XLS' },
  xls:  { bg: '#dcfce7', color: '#16a34a', label: 'XLS' },
  pdf:  { bg: '#fee2e2', color: '#dc2626', label: 'PDF' },
};

const THUMB_EXTS = new Set(['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff']);
const WIKI_PREFIX = 'Azure Wiki - ';

const isUrl = (name) => /^https?:\/\//i.test(name);
const humanize = (seg) => seg.replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim();

// Turn a raw indexed-source string into a readable title + a muted subtitle.
// Wiki sources look like "Azure Wiki - <project>/<wikiId>/<path…>", which is
// unreadable as-is, so we surface just the page title and its section path.
function parseDocName(name) {
  if (name.startsWith(WIKI_PREFIX)) {
    const segs = name.slice(WIKI_PREFIX.length).split('/').filter(Boolean);
    const pathSegs = segs.slice(2); // drop <project> and <wikiId>
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

export default function Sidebar({
  conversations,
  activeId,
  sessionId,
  indexedFiles,
  uploading,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onRenameChat,
  onClearAll,
  onUpload,
  onIngestWiki,
  onClearFiles,
  onDeleteFile,
}) {
  const [search, setSearch] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editingText, setEditingText] = useState('');
  const [wikiUrl, setWikiUrl] = useState('');
  const [wikiPat, setWikiPat] = useState('');
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
        <div className="docs-list">
          {indexedFiles.map((name) => {
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
                  title="Remove"
                  onClick={() => onDeleteFile(name)}
                >
                  ×
                </button>
              </div>
            );
          })}
          {indexedFiles.length === 0 && (
            <div className="docs-empty">No documents indexed yet.</div>
          )}
        </div>
      </div>
    </aside>
  );
}
