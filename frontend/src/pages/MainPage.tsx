import './MainPage.css';

import React, { useEffect, useState } from 'react';

import Button from '../components/Button';
import ModelCard from '../components/ModelCard';
import type { Point } from '../utils/generateGalaxyPoints';
import SliderInput from '../components/SliderInput'; // Import the new component
import api from '../services/api';
import generateGalaxyPoints from '../utils/generateGalaxyPoints';
import modelsData from '../utils/modelsData';

const MainPage: React.FC = () => {
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [trainingStatus, setTrainingStatus] = useState<string>(''); 
    const [galaxyPoints, setGalaxyPoints] = useState<Point[]>([]);

    const [turns, setTurns] = useState<number>(2);
    const [pointsPerTurn, setPointsPerTurn] = useState<number>(100);
    const [radiusStart, setRadiusStart] = useState<number>(0);
    const [radiusEnd, setRadiusEnd] = useState<number>(100);

    useEffect(() => {
        const points = generateGalaxyPoints(turns, pointsPerTurn, radiusStart, radiusEnd); 
        setGalaxyPoints(points);
    }, [turns, pointsPerTurn, radiusStart, radiusEnd]);

    const handleTrainModel = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        try {
            const response = await api.post('/models/train', {
                model_type: selectedModel,
                dataset: 'Galaxy Points', 
            });
            setTrainingStatus(`Training job queued successfully: ${response.data.job_id}`);
        } catch (error) {
            console.error('Error training model:', error);
            setTrainingStatus('Training failed');
        }
    };

    return (
        <div className="main-container">
            <h1>Train Toy Generative Models</h1>
            <p className="intro-text">
                Explore various generative models and train them on a galaxy-like distribution of points.
            </p>
            <h2>Models</h2>
            <div className="models-list">
                {modelsData.map((model) => (
                    <ModelCard 
                        key={model.type} 
                        title={model.type} 
                        description={model.description} 
                        onSelect={() => setSelectedModel(model.type)} 
                    />
                ))}
            </div>

            <h2>Dataset</h2>
            <p>The toy data distribution forms a spiral or a galaxy from 2D points.</p>
            <div className="parameter-controls">
                <SliderInput 
                    label="Turns" 
                    value={turns} 
                    min={1} 
                    max={5} 
                    onChange={setTurns} 
                />
                <SliderInput 
                    label="Radius Start" 
                    value={radiusStart} 
                    min={0} 
                    max={100} 
                    onChange={setRadiusStart} 
                />
                <SliderInput 
                    label="Radius End" 
                    value={radiusEnd} 
                    min={0} 
                    max={200} 
                    onChange={setRadiusEnd} 
                />
            </div>

            {galaxyPoints.length > 0 && (
                <div className="galaxy-preview">
                    <svg width="400" height="400" style={{ border: '1px solid white' }}>
                        {galaxyPoints.map((point, index) => (
                            <circle key={index} cx={point.x + 200} cy={point.y + 200} r="2" fill="#ffcc00" />
                        ))}
                    </svg>
                </div>
            )}

            <h2>Start Training</h2>
            <form onSubmit={handleTrainModel} className="training-form">
                <Button type="submit" disabled={!selectedModel} onClick={() => {}}>
                    Train Model
                </Button>
            </form>

            {trainingStatus && <p className="status-message">{trainingStatus}</p>}
        </div>
    );
};

export default MainPage;