import { useCallback, useEffect, useRef, useState } from 'react';
import Message from './Message.jsx';
import { ChevronDownIcon } from './Icons.jsx';

export default function ChatPane({ conversation }) {
  const endRef = useRef(null);
  const paneRef = useRef(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  const isNearBottom = useCallback(() => {
    const el = paneRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 300;
  }, []);

  useEffect(() => {
    if (!paneRef.current) return;
    if (isNearBottom()) {
      endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversation?.messages, isNearBottom]);

  const onScroll = useCallback(() => {
    setShowScrollBtn(!isNearBottom());
  }, [isNearBottom]);

  const scrollToBottom = () => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  if (!conversation || conversation.messages.length === 0) {
    return (
      <div className="chat-pane empty">
        <div className="empty-hero">
          <h1>DocPilot</h1>
          <p>Upload your PDFs, images, CSVs or spreadsheets and ask anything about them.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-pane" ref={paneRef} onScroll={onScroll}>
      {conversation.messages.map((m, i) => (
        <Message key={m.id ?? i} message={m} />
      ))}
      <div ref={endRef} />
      {showScrollBtn && (
        <button className="scroll-bottom-btn" onClick={scrollToBottom} aria-label="Scroll to bottom">
          <ChevronDownIcon />
        </button>
      )}
    </div>
  );
}
