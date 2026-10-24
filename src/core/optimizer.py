import numpy as np
from abc import ABC, abstractmethod
from src.core.layer import Layer

class Optimizer(ABC):
    @abstractmethod
    def step(self, layers: list[Layer]):
        pass

    @abstractmethod
    def zero_grad(self, layers: list[Layer]):
        pass

class SGD(Optimizer):
    def __init__(self, lr: float, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = {} 

    def step(self, layers: list[Layer]):
        for layer in layers:
            for p in layer.get_params():
                if p.grad is None:
                    continue
                pid = id(p)
                if pid not in self.v:
                    self.v[pid] = np.zeros_like(p.data)
                self.v[pid] = self.momentum * self.v[pid] + p.grad
                p.data -= self.lr * self.v[pid]


    def zero_grad(self, layers: list[Layer]):
        for layer in layers:
            for p in layer.get_params():
                p.zero_grad()

