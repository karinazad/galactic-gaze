import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';

import MainPage from './pages/MainPage';
import React from 'react';

const App: React.FC = () => {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<MainPage />} /> {/* Set MainPage as the default route */}
                {}
            </Routes>
        </Router>
    );
};

export default App;