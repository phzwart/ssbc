"""Simulation utilities for testing conformal prediction."""

import numpy as np


class BinaryClassifierSimulator:
    """Simulate binary classification data with probabilities from Beta distributions.

    This simulator generates realistic classification scenarios where the predicted
    probabilities for each class follow Beta distributions. Useful for testing and
    benchmarking conformal prediction methods.

    Parameters
    ----------
    p_class1 : float
        Probability of drawing class 1 (class imbalance parameter)
        Must be in [0, 1]
    beta_params_class0 : tuple of (a, b)
        Beta distribution parameters for p(class=1) when true label is 0
        Typically use parameters that give low probabilities (e.g., (2, 8))
    beta_params_class1 : tuple of (a, b)
        Beta distribution parameters for p(class=1) when true label is 1
        Typically use parameters that give high probabilities (e.g., (8, 2))
    seed : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    p_class1 : float
        Probability of class 1
    p_class0 : float
        Probability of class 0 (= 1 - p_class1)
    a0, b0 : float
        Beta parameters for class 0
    a1, b1 : float
        Beta parameters for class 1
    rng : numpy.random.Generator
        Random number generator

    Examples
    --------
    >>> # Simulate imbalanced data: 10% positive class
    >>> # Class 0: Beta(2, 8) → mean p(class=1) = 0.2 (low scores, correct)
    >>> # Class 1: Beta(8, 2) → mean p(class=1) = 0.8 (high scores, correct)
    >>> sim = BinaryClassifierSimulator(
    ...     p_class1=0.10,
    ...     beta_params_class0=(2, 8),
    ...     beta_params_class1=(8, 2),
    ...     seed=42
    ... )
    >>> labels, probs = sim.generate(n_samples=100)
    >>> print(labels.shape)
    (100,)
    >>> print(probs.shape)
    (100, 2)

    Notes
    -----
    The Beta distribution parameters (a, b) control the shape:
    - Mean = a / (a + b)
    - For a classifier that works well:
      - Class 0 should have low p(class=1): use (a, b) with a < b
      - Class 1 should have high p(class=1): use (a, b) with a > b
    """

    def __init__(
        self,
        p_class1: float,
        beta_params_class0: tuple[float, float],
        beta_params_class1: tuple[float, float],
        seed: int | None = None,
    ):
        """Initialize the binary classifier simulator."""
        if not 0 <= p_class1 <= 1:
            raise ValueError("p_class1 must be in [0, 1]")

        self.p_class1 = p_class1
        self.p_class0 = 1.0 - p_class1
        self.a0, self.b0 = beta_params_class0
        self.a1, self.b1 = beta_params_class1
        self.rng = np.random.default_rng(seed)

        # Validate beta parameters
        if self.a0 <= 0 or self.b0 <= 0:
            raise ValueError("Beta parameters for class 0 must be positive")
        if self.a1 <= 0 or self.b1 <= 0:
            raise ValueError("Beta parameters for class 1 must be positive")

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate n_samples of (label, p(class=0), p(class=1)).

        Parameters
        ----------
        n_samples : int
            Number of samples to generate

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)
            True binary labels (0 or 1)
        probs : np.ndarray, shape (n_samples, 2)
            Classification probabilities [p(class=0), p(class=1)]
            Each row sums to 1.0

        Examples
        --------
        >>> sim = BinaryClassifierSimulator(
        ...     p_class1=0.5,
        ...     beta_params_class0=(2, 8),
        ...     beta_params_class1=(8, 2),
        ...     seed=42
        ... )
        >>> labels, probs = sim.generate(n_samples=5)
        >>> print(f"Generated {len(labels)} samples")
        Generated 5 samples
        >>> print(f"Class balance: {np.bincount(labels)}")
        Class balance: [2 3]
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        # Draw true labels according to class distribution
        labels = self.rng.choice([0, 1], size=n_samples, p=[self.p_class0, self.p_class1])

        # Initialize probability array
        probs = np.zeros((n_samples, 2))

        # For each label, draw classification probability from appropriate Beta
        for i, label in enumerate(labels):
            if label == 0:
                # True label is 0: sample p(class=1) from Beta(a0, b0)
                p_class1 = self.rng.beta(self.a0, self.b0)
            else:
                # True label is 1: sample p(class=1) from Beta(a1, b1)
                p_class1 = self.rng.beta(self.a1, self.b1)

            probs[i, 1] = p_class1  # p(class=1)
            probs[i, 0] = 1.0 - p_class1  # p(class=0)

        return labels, probs

    def __repr__(self) -> str:
        """String representation of the simulator."""
        return (
            f"BinaryClassifierSimulator(p_class1={self.p_class1:.3f}, "
            f"beta_class0=({self.a0}, {self.b0}), "
            f"beta_class1=({self.a1}, {self.b1}))"
        )
