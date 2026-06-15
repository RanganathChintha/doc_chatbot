import { useEffect, useRef } from 'react';

export default function Toast({ message, type = 'error', onClose }) {
  const timerRef = useRef(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!message) return;
    timerRef.current = setTimeout(() => onCloseRef.current(), 5000);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [message]);

  if (!message) return null;

  return (
    <div className={`toast toast-${type}`} role="alert">
      <span className="toast-text">{message}</span>
      <button className="toast-close" onClick={onClose} aria-label="Dismiss">&times;</button>
    </div>
  );
}
