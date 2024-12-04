import './Spinner.css';

import React from 'react';

const Spinner: React.FC = () => {
    return (
        <div className="spinner">
            {/* Add a loading spinner here */}
            <div className="loading-spinner"></div>
        </div>
    );
};

export default Spinner;