import { memo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { StarIcon, CopyIcon, CheckIcon, RefreshIcon } from './Icons.jsx';
import CodeBlock from './CodeBlock.jsx';

const MARKDOWN_COMPONENTS = { pre: CodeBlock };

const Message = memo(function Message({ message, isLast, streaming, onRegenerate }) {
  const isUser = message.role === 'user';
  const [showSources, setShowSources] = useState(false);
  const [copied, setCopied] = useState(false);

  const sources = message.sources || [];
  const hasSources = !isUser && sources.length > 0;

  const copyMessage = async () => {
    try {
      await navigator.clipboard.writeText(message.content || '');
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  const sourceLabel = (s) => s.title || (s.source || 'unknown').split(/[\\/]/).pop();
  const formatScore = (score) => {
    const pct = Math.round((score > 1 ? score / 100 : score) * 100);
    return `${Math.max(0, Math.min(100, pct))}% match`;
  };

  return (
    <div className={`msg-row ${isUser ? 'user' : 'assistant'}`}>
      <div className="msg-avatar">{isUser ? 'U' : <StarIcon />}</div>
      <div className="msg-body">
        {!isUser && <div className="msg-label">DocPilot</div>}
        <div className="msg-content">
          {message.content ? (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={MARKDOWN_COMPONENTS}
            >
              {message.content}
            </ReactMarkdown>
          ) : (
            <span className="typing-dot">●</span>
          )}
          {message.stopped && <div className="msg-stopped">— Stopped —</div>}
        </div>

        {!isUser && message.content && !streaming && (
          <div className="msg-actions">
            <button className="msg-action-btn" onClick={copyMessage} aria-label="Copy response">
              {copied ? <CheckIcon /> : <CopyIcon />}
              {copied ? 'Copied' : 'Copy'}
            </button>
            {isLast && onRegenerate && (
              <button className="msg-action-btn" onClick={onRegenerate} aria-label="Regenerate response">
                <RefreshIcon />
                Regenerate
              </button>
            )}
          </div>
        )}

        {hasSources && (
          <div className="msg-sources">
            <button className="sources-toggle" onClick={() => setShowSources((v) => !v)}>
              {showSources ? '▾' : '▸'} Sources used ({sources.length})
            </button>
            {showSources && (
              <ul className="sources-list">
                {sources.map((s, i) => (
                  <li key={i} className="source-item">
                    <div className="source-head">
                      <span className="source-num">{i + 1}</span>
                      <span className="source-tag">{(s.source_type || '?').toUpperCase()}</span>
                      {s.url ? (
                        <a className="source-name" href={s.url} target="_blank" rel="noopener noreferrer">{sourceLabel(s)}</a>
                      ) : (
                        <span className="source-name">{sourceLabel(s)}</span>
                      )}
                      {s.page != null && <span className="source-page">p.{s.page}</span>}
                      {s.score != null && <span className="source-score">{formatScore(s.score)}</span>}
                    </div>
                    <div className="source-snippet">{s.snippet}{s.truncated ? '…' : ''}</div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
});

export default Message;
