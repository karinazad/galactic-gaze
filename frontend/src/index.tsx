// src/index.tsx

import App from './App';
import React from 'react';
import ReactDOM from 'react-dom/client';
// import reportWebVitals from './reportWebVitals'; // Comment this out if not needed

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);

// Uncomment if you want to use reportWebVitals
// reportWebVitals();