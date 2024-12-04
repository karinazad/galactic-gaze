import React from 'react';

interface ButtonProps {
    onClick: () => void; 
    children: React.ReactNode;
    disabled?: boolean;
    type?: "button" | "submit" | "reset"; 
}

const Button: React.FC<ButtonProps> = ({ onClick, children, disabled = false, type = "button" }) => {
    return (
        <button onClick={onClick} disabled={disabled} className="custom-button" type={type}>
            {children}
        </button>
    );
};

export default Button;