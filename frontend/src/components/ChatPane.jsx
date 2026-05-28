import { useEffect, useRef } from 'react';
import Message from './Message.jsx';

export default function ChatPane({ conversation }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation?.messages]);

  if (!conversation || conversation.messages.length === 0) {
    return (
      <div className="chat-pane empty">
        <div className="empty-hero">
          <h1>CHAT A.I+</h1>
          <p>Upload your PDFs, images, CSVs or spreadsheets and ask anything about them.</p>
          <ul className="suggestion-grid">
            <li>Summarize this document</li>
            <li>List all key entities mentioned</li>
            <li>Compare findings across pages</li>
            <li>Explain the chart on page 3</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-pane">
      {conversation.messages.map((m, i) => (
        <Message key={i} message={m} />
      ))}
      <div ref={endRef} />
    </div>
  );
}
