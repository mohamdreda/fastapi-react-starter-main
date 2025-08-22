import React from 'react'
import ReactDOM from 'react-dom/client'

// Apply persisted theme early
const persistedTheme = localStorage.getItem('theme')
if (persistedTheme === 'dark') {
  document.documentElement.classList.add('dark')
}
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
