import numpy as np
from abc import ABC, abstractmethod
from src.core.parameter import Parameter

class Layer(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    def get_params(self) -> list:
        return []

class LinearLayer(Layer):
    def __init__(self, input_size: int, output_size: int, enable_bias: bool = True):
        self.input_size = input_size
        self.output_size = output_size
        self.enable_bias = enable_bias

        w_data = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.weights = Parameter(w_data, name="weights")

        if self.enable_bias:
            b_data = np.zeros((1, output_size))
            self.bias = Parameter(b_data, name="bias")
        else:
            self.bias = None

        self.X = None

    def get_params(self) -> list:
        params = [self.weights]
        if self.bias is not None:
            params.append(self.bias)
        return params

    @property
    def dW(self):
        return self.weights.grad

    @property
    def db(self):
        return self.bias.grad if self.bias is not None else None

    def forward(self, X):
        # X - must be horizontal
        self.X = np.copy(X)
        
        with_weights = self.X @ self.weights.data
        if self.bias is not None:
            return with_weights + self.bias.data
        
        return with_weights

    def backward(self, grad_output):
        if self.X is None:
            raise ValueError("forward was not called")

        # Gradients of weights and biases
        self.weights.grad = self.X.T @ grad_output
        if self.bias is not None:
            self.bias.grad = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient for input
        return grad_output @ self.weights.data.T
