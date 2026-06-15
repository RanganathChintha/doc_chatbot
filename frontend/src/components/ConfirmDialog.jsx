import { useEffect, useRef } from 'react';

export default function ConfirmDialog({ open, title, message, confirmLabel = 'Confirm', cancelLabel = 'Cancel', danger = false, onConfirm, onCancel }) {
  const confirmRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    confirmRef.current?.focus();
    const onKey = (e) => {
      if (e.key === 'Escape') onCancel();
      if (e.key === 'Enter') onConfirm();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onConfirm, onCancel]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onMouseDown={onCancel}>
      <div
        className="modal"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        onMouseDown={(e) => e.stopPropagation()}
      >
        {title && <h2 id="confirm-title" className="modal-title">{title}</h2>}
        <p className="modal-message">{message}</p>
        <div className="modal-actions">
          <button className="modal-btn cancel" onClick={onCancel}>{cancelLabel}</button>
          <button
            ref={confirmRef}
            className={`modal-btn confirm ${danger ? 'danger' : ''}`}
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
