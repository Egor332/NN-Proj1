import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.layer import LinearLayer
from src.core.activations import CustomReLU, Linear
from src.core.loss import MSELoss
from src.core.optimizer import SGD
from src.core.network import MLP, set_seed
from src.utils.data_loader import load_dataset, StandardScaler

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_DIM = 32
LR = 0.005
MOMENTUM = 0.9
SEED = 42

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

def get_model_params(model: MLP) -> list[np.ndarray]:
    """Extracts copies of all trainable parameters (weights and biases)."""
    return [np.copy(p.data) for layer in model.layers for p in layer.get_params()]

def compute_total_weight_norm(params: list[np.ndarray]) -> float:
    """Computes total Frobenius/L2 norm of all parameters: ||W||_F."""
    sq_sum = sum(float(np.sum(p ** 2)) for p in params)
    return float(np.sqrt(sq_sum))

def compute_step_magnitude(prev_params: list[np.ndarray], curr_params: list[np.ndarray]) -> float:
    """Computes parameter update step magnitude between consecutive epochs: ||W(t) - W(t-1)||."""
    sq_sum = sum(float(np.sum((c - p) ** 2)) for p, c in zip(prev_params, curr_params))
    return float(np.sqrt(sq_sum))

def run_single_model_experiment():
    print("\n" + "=" * 70)
    print("  RUNNING REGRESSION DYNAMICS EXPERIMENT (1 Hidden Layer, ReLU, MSE)  ")
    print(f"  Dataset: cube (1000 samples) | Epochs: {EPOCHS} | Seed: {SEED}     ")
    print("=" * 70)

    # 1. Load data
    X_train, y_train = load_dataset("regression", "cube", 1000, split="train")
    X_test, y_test = load_dataset("regression", "cube", 1000, split="test")

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_scaled = scaler_X.transform(X_train)
    y_train_scaled = scaler_y.transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)

    # 2. Build model: 1 hidden layer (32 neurons) with ReLU and MSE Loss
    set_seed(SEED)
    layers = [
        LinearLayer(input_size=1, output_size=HIDDEN_DIM, enable_bias=True),
        CustomReLU(),
        LinearLayer(input_size=HIDDEN_DIM, output_size=1, enable_bias=True),
        Linear()
    ]
    optimizer = SGD(lr=LR, momentum=MOMENTUM)
    model = MLP(layers=layers, optimizer=optimizer, loss_fn=MSELoss(), seed=SEED)

    # 3. Track metrics over epochs
    train_loss_history = []
    test_loss_history = []
    weight_norm_history = []
    weight_step_history = []

    # Initial state (epoch 0)
    initial_params = get_model_params(model)
    prev_params = initial_params

    # Evaluate test loss on unscaled targets
    def evaluate_test_mse(m: MLP) -> float:
        preds_scaled = m.predict(X_test_scaled)
        preds_real = scaler_y.inverse_transform(preds_scaled)
        return float(np.mean((y_test - preds_real) ** 2))

    init_test_mse = evaluate_test_mse(model)
    test_loss_history.append(init_test_mse)
    weight_norm_history.append(compute_total_weight_norm(initial_params))

    # Evaluate initial train loss on real targets
    init_train_preds = scaler_y.inverse_transform(model.predict(X_train_scaled))
    init_train_mse = float(np.mean((y_train - init_train_preds) ** 2))
    train_loss_history.append(init_train_mse)

    # Callback after each epoch
    def on_epoch_end(epoch: int, m: MLP):
        nonlocal prev_params
        curr_params = get_model_params(m)

        # Train loss of current epoch
        train_preds_real = scaler_y.inverse_transform(m.predict(X_train_scaled))
        current_train_mse = float(np.mean((y_train - train_preds_real) ** 2))
        train_loss_history.append(current_train_mse)

        # Test loss
        current_test_mse = evaluate_test_mse(m)
        test_loss_history.append(current_test_mse)

        # Weight norm
        w_norm = compute_total_weight_norm(curr_params)
        weight_norm_history.append(w_norm)

        # Weight step magnitude
        step_mag = compute_step_magnitude(prev_params, curr_params)
        weight_step_history.append(step_mag)

        prev_params = curr_params

        print(f"Epoch {epoch:2d}/{EPOCHS} -> Train MSE: {current_train_mse:10.2f} | Test MSE: {current_test_mse:10.2f} | ||W||: {w_norm:6.3f} | ||ΔW||: {step_mag:6.4f}")

    # 4. Train the model
    model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED,
        eval_every=1,
        eval_callback=on_epoch_end
    )

    # 5. Generate 2x2 Plot
    print("\nGenerating Visualisation (2x2 Grid)...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Training Dynamics: 1-Hidden-Layer MLP on cube 1000 (MSE Loss, ReLU)", fontsize=15, fontweight="bold", y=0.98)

    epochs_with_zero = np.arange(0, EPOCHS + 1)
    step_epochs = np.arange(1, EPOCHS + 1)

    # Subplot 1: Train Loss Reduction
    ax1 = axes[0, 0]
    ax1.plot(epochs_with_zero, train_loss_history, color="#1f77b4", linewidth=2.4, marker="o", markersize=4, label="Train Loss (MSE)")
    ax1.set_title("1. Train Loss Reduction over Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Train MSE", fontweight="bold", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Subplot 2: Test Loss Reduction
    ax2 = axes[0, 1]
    ax2.plot(epochs_with_zero, test_loss_history, color="#d95f02", linewidth=2.4, marker="s", markersize=4, label="Test Loss (MSE)")
    ax2.set_title("2. Test Loss Reduction over Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Test MSE", fontweight="bold", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9)

    # Subplot 3: Weights Norm
    ax3 = axes[1, 0]
    ax3.plot(epochs_with_zero, weight_norm_history, color="#2ca02c", linewidth=2.4, marker="^", markersize=4, label="Total Weight Norm ||W||")
    ax3.set_title("3. Weights Norm ||W|| over Epochs", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch", fontweight="bold", fontsize=11)
    ax3.set_ylabel("Weight Norm ||W||", fontweight="bold", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower right", framealpha=0.9)

    # Subplot 4: Weights Step Magnitude
    ax4 = axes[1, 1]
    ax4.plot(step_epochs, weight_step_history, color="#9467bd", linewidth=2.4, marker="D", markersize=4, label="Step Magnitude ||W(t) - W(t-1)||")
    ax4.set_title("4. Weights Step Magnitude ||W(t) - W(t-1)|| over Epochs", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Epoch", fontweight="bold", fontsize=11)
    ax4.set_ylabel("Step Magnitude ||W(t) - W(t-1)||", fontweight="bold", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    output_path = RESULTS_DIR / "regression_single_run_dynamics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSuccessfully saved plot to: {output_path}")

if __name__ == "__main__":
    run_single_model_experiment()
