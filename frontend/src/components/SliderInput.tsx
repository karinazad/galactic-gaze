import './SliderInput.css';

// SliderInput.tsx
import React from 'react';

interface SliderInputProps {
    label: string;
    value: number;
    min: number;
    max: number;
    onChange: (value: number) => void;
}

const SliderInput: React.FC<SliderInputProps> = ({ label, value, min, max, onChange }) => {
    return (
        <div className="slider-input">
            <label>
                {label}:
                <input
                    type="range"
                    min={min}
                    max={max}
                    value={value}
                    onChange={(e) => onChange(Number(e.target.value))}
                    className="slider"
                />
                <span className="slider-value">{value}</span>
            </label>
        </div>
    );
};

export default SliderInput;