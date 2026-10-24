import numpy as np
from abc import ABC, abstractmethod
from src.core.layer import Layer

class ActivationFunction(Layer):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, grad_output):
        pass

class CustomReLU(ActivationFunction):
    def __init__(self):
        self.X = None
        
    def forward(self, x):
        self.X = np.copy(x)
        return np.maximum(x, 0)

    def backward(self, grad_output):
        if self.X is None:
            raise ValueError("X is None")

        drelu = np.where(self.X > 0, 1.0, 0.0)
        return drelu * grad_output

class Sigmoid(ActivationFunction):
    def __init__(self):
        self.out = None

    def forward(self, x):
        x_clipped = np.clip(x, -500.0, 500.0)
        self.out = 1.0 / (1.0 + np.exp(-x_clipped))
        return self.out

    def backward(self, grad_output):
        if self.out is None:
            raise ValueError("forward was not called before backward")
        return grad_output * self.out * (1.0 - self.out)

class Tanh(ActivationFunction):
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_output):
        if self.out is None:
            raise ValueError("forward was not called before backward")
        return grad_output * (1.0 - self.out ** 2)

class Linear(ActivationFunction):
    def forward(self, x):
        return x

    def backward(self, grad_output):
        return grad_output