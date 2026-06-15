import { memo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { StarIcon } from './Icons.jsx';

const Message = memo(function Message({ message }) {
  const isUser = message.role === 'user';
  const [showSources, setShowSources] = useState(false);

  return (
    <div className={`msg-row ${isUser ? 'user' : 'assistant'}`}>
      <div className="msg-avatar">{isUser ? 'U' : <StarIcon />}</div>
      <div className="msg-body">
        {!isUser && <div className="msg-label">DocPilot</div>}
        <div className="msg-content">
          {message.content ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          ) : (
            <span className="typing-dot">●</span>
          )}
          {message.stopped && <div className="msg-stopped">— Stopped —</div>}
        </div>
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="msg-sources">
            <button className="sources-toggle" onClick={() => setShowSources((v) => !v)}>
              {showSources ? '▾' : '▸'} Sources used ({message.sources.length})
            </button>
            {showSources && (
              <ul className="sources-list">
                {message.sources.map((s, i) => {
                  const label = s.title || (s.source || 'unknown').split(/[\\/]/).pop();
                  return (
                    <li key={i} className="source-item">
                      <div className="source-head">
                        <span className="source-tag">{(s.source_type || '?').toUpperCase()}</span>
                        {s.url ? (
                          <a className="source-name" href={s.url} target="_blank" rel="noopener noreferrer">{label}</a>
                        ) : (
                          <span className="source-name">{label}</span>
                        )}
                        {s.page != null && <span className="source-page">p.{s.page}</span>}
                      </div>
                      <div className="source-snippet">{s.snippet}{s.truncated ? '…' : ''}</div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
});

export default Message;
