import math
from typing import List, Tuple


def generate_galaxy_points(
    turns: int = 2,
    points_per_turn: int = 10000,
    radius_start: float = 0,
    radius_end: float = 100,
) -> List[Tuple[float, float]]:
    """
    Parameters
    ----------
    turns : int, optional
        Number of spiral turns, by default 2
    points_per_turn : int, optional
        Number of points generated in each turn, by default 100
    radius_start : float, optional
        Starting radius of the spiral, by default 0
    radius_end : float, optional
        Ending radius of the spiral, by default 100

    Returns
    -------
    List[Tuple[float, float]]
        List of (x, y) coordinate points forming a spiral
    """
    points = []
    total_points = turns * points_per_turn

    for i in range(total_points):
        theta = (i / points_per_turn) * (2 * math.pi * turns)
        radius = radius_start + (radius_end - radius_start) * (i / total_points)

        x = radius * math.cos(theta)
        y = radius * math.sin(theta)

        points.append((x, y))

    return points
