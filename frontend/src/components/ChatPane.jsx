import { useCallback, useEffect, useRef, useState } from 'react';
import Message from './Message.jsx';
import { ChevronDownIcon } from './Icons.jsx';

const SUGGESTED_PROMPTS = [
  'Summarize the key points of this document.',
  'What are the main takeaways?',
  'List any action items or deadlines mentioned.',
  'Are there any risks or open questions?',
];

export default function ChatPane({ conversation, streaming, hasDocs, onSend, onRegenerate }) {
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
          {hasDocs && (
            <div className="suggested-prompts">
              {SUGGESTED_PROMPTS.map((p) => (
                <button key={p} className="suggested-prompt" onClick={() => onSend?.(p)}>
                  {p}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  const lastAssistantIdx = (() => {
    const msgs = conversation.messages;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === 'assistant') return i;
    }
    return -1;
  })();

  return (
    <div className="chat-pane" ref={paneRef} onScroll={onScroll}>
      {conversation.messages.map((m, i) => (
        <Message
          key={m.id ?? i}
          message={m}
          isLast={i === lastAssistantIdx}
          streaming={streaming}
          onRegenerate={onRegenerate}
        />
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
