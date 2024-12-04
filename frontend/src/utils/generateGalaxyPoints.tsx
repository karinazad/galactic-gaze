interface Point {
    x: number;
    y: number;
}

const generateGalaxyPoints = (
    turns: number = 2,
    pointsPerTurn: number = 100,
    radiusStart: number = 0,
    radiusEnd: number = 100
): Point[] => {
    const points: Point[] = [];
    const totalPoints = turns * pointsPerTurn;
    
    for (let i = 0; i < totalPoints; i++) {
        const theta = (i / pointsPerTurn) * (2 * Math.PI * turns);
        const radius = radiusStart + (radiusEnd - radiusStart) * (i / totalPoints);
        
        const x = radius * Math.cos(theta);
        const y = radius * Math.sin(theta);

        points.push({ x, y });
    }

    return points;
};

export default generateGalaxyPoints;
export type { Point };
