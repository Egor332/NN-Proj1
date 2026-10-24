import numpy as np
from src.core.layer import Layer
from src.core.loss import Loss
from src.core.optimizer import Optimizer

def set_seed(seed: int):
    """Set global random seed for numpy to ensure reproducible training."""
    np.random.seed(seed)

class MLP:
    def __init__(self, layers: list[Layer], optimizer: Optimizer, loss_fn: Loss, seed: int | None = None):
        if seed is not None:
            set_seed(seed)
        self.layers = layers
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.seed = seed

        self.history = {
            "loss": [],
            "weight_snapshot": []
        }

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, grad_output):
        back_grad = grad_output
        for layer in reversed(self.layers):
            back_grad = layer.backward(back_grad)
        
    def fit(
        self,
        X,
        y,
        epochs,
        batch_size: int = 32,
        record_weights: int = 100,
        seed: int | None = None,
        eval_every: int | None = None,
        eval_callback = None
    ):
        if seed is not None:
            set_seed(seed)
        elif self.seed is not None:
            set_seed(self.seed)

        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        train_size = X.shape[0]

        for epoch in range(epochs):
            shufled_indicies = np.random.permutation(train_size)
            X_shufled = X[shufled_indicies]
            Y_shufled = y[shufled_indicies]

            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, train_size, batch_size):
                x_batch = X_shufled[i:i+batch_size, :]
                y_batch = Y_shufled[i:i+batch_size, :]

                x_batch_prediction = self.forward(x_batch)
                loss = self.loss_fn.forward(y_batch, x_batch_prediction)
                initial_grad = self.loss_fn.backward()
                self.backward(initial_grad)
                
                self.optimizer.step(self.layers)
                self.optimizer.zero_grad(self.layers)
                
                epoch_loss += loss
                num_batches += 1

            self.history['loss'].append(epoch_loss / num_batches)
            if (epoch % record_weights == 0) or (epoch == epochs - 1):
                snapshot = [np.copy(p.data) for layer in self.layers for p in layer.get_params()]
                self.history['weight_snapshot'].append((epoch, snapshot))

            if eval_callback is not None and eval_every is not None:
                if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
                    eval_callback(epoch + 1, self)

    def predict(self, X):
        return self.forward(X)