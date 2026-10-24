import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def forward(self, y_true, y_pred):
        pass

    @abstractmethod
    def backward(self):
        pass

class MSELoss(Loss):
    def __init__(self):
        self.y_true = None
        self.y_pred = None
    
    def forward(self, y_true, y_pred):
        self.y_true = np.copy(y_true)
        self.y_pred = np.copy(y_pred)
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self):
        return (2 / self.y_pred.size) * (self.y_pred - self.y_true)

class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self):
        self.y_true = None
        self.logits = None
    
    def forward(self, y_true, logits):
        # assumes y_true is [0, 0, 0, 1, 0], ... not just [3] (index)
        self.y_true = np.copy(y_true)
        
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True) # for numerical stability
        exp_logits = np.exp(shifted_logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        self.probabilities = np.copy(probabilities)

        clipped_probs = np.clip(self.probabilities, 1e-15, 1.0 - 1e-15)

        N = y_true.shape[0] # batch size
        loss = -np.sum(y_true * np.log(clipped_probs)) / N
        return loss
    
    def backward(self):
        N = self.y_true.shape[0]    
        return (self.probabilities - self.y_true) / N

class MAELoss(Loss):
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_true, y_pred):
        self.y_true = np.copy(y_true)
        self.y_pred = np.copy(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    def backward(self):
        if self.y_true is None or self.y_pred is None:
            raise ValueError("forward was not called before backward")
        return np.sign(self.y_pred - self.y_true) / self.y_pred.size

class BinaryCrossEntropyLoss(Loss):
    def __init__(self, eps: float = 1e-15):
        self.y_true = None
        self.y_pred = None
        self.eps = eps

    def forward(self, y_true, y_pred):
        self.y_true = np.copy(y_true)
        self.y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss = -np.mean(self.y_true * np.log(self.y_pred) + (1.0 - self.y_true) * np.log(1.0 - self.y_pred))
        return loss

    def backward(self):
        if self.y_true is None or self.y_pred is None:
            raise ValueError("forward was not called before backward")
        return ((self.y_pred - self.y_true) / (self.y_pred * (1.0 - self.y_pred))) / self.y_pred.size