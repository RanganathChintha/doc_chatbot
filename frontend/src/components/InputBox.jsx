import { useEffect, useRef, useState } from 'react';
import { SendIcon, StopIcon } from './Icons.jsx';

const MAX_CHARS = 4000;

export default function InputBox({ disabled, disabledReason, streaming, onSend, onStop, placeholder }) {
  const [value, setValue] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, [value]);

  const submit = () => {
    if (!value.trim() || streaming) return;
    onSend(value);
    setValue('');
  };

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const overLimit = value.length > MAX_CHARS;
  const showCounter = value.length > MAX_CHARS * 0.75;

  return (
    <div className="input-wrap">
      <div className={`input-box ${disabled ? 'is-disabled' : ''}`} title={disabled ? disabledReason : undefined}>
        <textarea
          ref={textareaRef}
          rows={1}
          className="input-textarea"
          placeholder={placeholder}
          value={value}
          disabled={disabled}
          maxLength={MAX_CHARS + 200}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
        />
        {streaming ? (
          <button className="send-btn stop" onClick={onStop} aria-label="Stop generating">
            <StopIcon />
          </button>
        ) : (
          <button
            className="send-btn"
            onClick={submit}
            disabled={disabled || !value.trim() || overLimit}
            aria-label="Send message"
          >
            <SendIcon />
          </button>
        )}
      </div>
      <div className="input-footer">
        <span className="input-hint">
          {disabled && disabledReason ? disabledReason : 'Enter to send · Shift + Enter for a new line'}
        </span>
        {showCounter && (
          <span className={`input-counter ${overLimit ? 'over' : ''}`}>
            {value.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
          </span>
        )}
      </div>
    </div>
  );
}
