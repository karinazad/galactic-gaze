from typing import List, Tuple


def generate_moon_points(
    n_samples: int = 200, noise: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Parameters
    ----------
    n_samples : int, optional
        Total number of points to generate, by default 200
    noise : float, optional
        Standard deviation of Gaussian noise added to points, by default 0.1

    Returns
    -------
    List[Tuple[float, float]]
        List of (x, y) coordinate points forming moon shapes
    """
    from sklearn.datasets import make_moons

    X, _ = make_moons(n_samples=n_samples, noise=noise)
    return X.tolist()
