import numpy as np

class Parameter:
    """
    Encapsulates a trainable model parameter (e.g., weights, biases)
    along with its associated gradient.
    """
    def __init__(self, data: np.ndarray, name: str = ""):
        self.data = np.asarray(data)
        self.grad = np.zeros_like(self.data)
        self.name = name

    def zero_grad(self):
        """Reset parameter gradient to zeros."""
        self.grad.fill(0.0)

    def __repr__(self):
        return f"Parameter(name='{self.name}', shape={self.data.shape})"
