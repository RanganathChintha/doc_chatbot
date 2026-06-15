import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './styles/variables.css';
import './styles/base.css';
import './styles/sidebar.css';
import './styles/chat.css';
import './styles/input.css';
import './styles/toast.css';
import './styles/modal.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
