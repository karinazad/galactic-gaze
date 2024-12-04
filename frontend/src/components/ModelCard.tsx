import React from 'react';

interface ModelCardProps {
    title: string;
    description: string;
    onSelect: () => void;
}

const ModelCard: React.FC<ModelCardProps> = ({ title, description, onSelect }) => {
    return (
        <div className="model-card">
            <h2>{title}</h2>
            <p>{description}</p>
            <button className="select-button" onClick={onSelect}>
                Select this Model
            </button>
        </div>
    );
};

export default ModelCard;