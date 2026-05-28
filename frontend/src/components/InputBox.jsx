import { useEffect, useRef, useState } from 'react';

export default function InputBox({ disabled, streaming, onSend, onStop, placeholder }) {
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

  return (
    <div className="input-box">
      <textarea
        ref={textareaRef}
        rows={1}
        className="input-textarea"
        placeholder={placeholder}
        value={value}
        disabled={disabled}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
      />
      {streaming ? (
        <button className="send-btn stop" onClick={onStop} title="Stop">■</button>
      ) : (
        <button
          className="send-btn"
          onClick={submit}
          disabled={disabled || !value.trim()}
          title="Send (Enter)"
        >
          ➤
        </button>
      )}
    </div>
  );
}
