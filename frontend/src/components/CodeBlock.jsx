import { useRef, useState } from 'react';
import { CopyIcon, CheckIcon } from './Icons.jsx';

// Custom renderer for fenced code blocks: keeps rehype-highlight's tokens
// (it styles the <code> child) and adds a language label + copy button.
export default function CodeBlock({ children }) {
  const preRef = useRef(null);
  const [copied, setCopied] = useState(false);

  const codeEl = Array.isArray(children) ? children[0] : children;
  const className = codeEl?.props?.className || '';
  const lang = (className.match(/language-([\w-]+)/) || [])[1] || '';

  const copy = async () => {
    const text = preRef.current?.textContent ?? '';
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className="code-block">
      <div className="code-block-head">
        <span className="code-lang">{lang || 'text'}</span>
        <button className="code-copy" onClick={copy} aria-label="Copy code">
          {copied ? <CheckIcon /> : <CopyIcon />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre ref={preRef}>{children}</pre>
    </div>
  );
}
