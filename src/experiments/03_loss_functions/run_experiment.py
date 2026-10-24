import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.layer import LinearLayer
from src.core.activations import CustomReLU, Linear, Sigmoid
from src.core.loss import MSELoss, MAELoss, BinaryCrossEntropyLoss, SoftmaxCrossEntropyLoss
from src.core.optimizer import SGD
from src.core.network import MLP, set_seed
from src.utils.data_loader import load_dataset, StandardScaler, to_one_hot

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
SEEDS = [42, 123, 999]
REG_EPOCHS = 40
CLS_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_DIMS = [32, 32]
REG_LR = 0.002
CLS_LR = 0.01
MOMENTUM = 0.9

# Modern Plot Styling
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
    """Extracts a copy of all trainable parameters (weights and biases) from the network."""
    return [np.copy(p.data) for layer in model.layers for p in layer.get_params()]

def compute_step_magnitude(prev_params: list[np.ndarray], curr_params: list[np.ndarray]) -> float:
    """
    Computes total parameter update step magnitude as the Euclidean norm (Frobenius norm)
    of the difference vector between consecutive epochs: ||theta_t - theta_{t-1}||_2
    """
    sq_sum = 0.0
    for p_prev, p_curr in zip(prev_params, curr_params):
        diff = p_curr - p_prev
        sq_sum += float(np.sum(diff ** 2))
    return float(np.sqrt(sq_sum))

def build_regression_model(loss_name: str, seed: int) -> MLP:
    """Builds a regression MLP [1 -> 32 -> 32 -> 1] with the specified loss function."""
    set_seed(seed)
    layers = [
        LinearLayer(input_size=1, output_size=HIDDEN_DIMS[0], enable_bias=True),
        CustomReLU(),
        LinearLayer(input_size=HIDDEN_DIMS[0], output_size=HIDDEN_DIMS[1], enable_bias=True),
        CustomReLU(),
        LinearLayer(input_size=HIDDEN_DIMS[1], output_size=1, enable_bias=True),
        Linear()
    ]
    loss_fn = MSELoss() if loss_name.upper() == "MSE" else MAELoss()
    optimizer = SGD(lr=REG_LR, momentum=MOMENTUM)
    return MLP(layers=layers, optimizer=optimizer, loss_fn=loss_fn, seed=seed)

def build_classification_model(loss_name: str, seed: int) -> MLP:
    """
    Builds a classification MLP [2 -> 32 -> 32 -> output] with the specified loss function.
    - BCE: 1 output unit + Sigmoid activation + BinaryCrossEntropyLoss
    - SoftmaxCE: 2 output units (logits) + SoftmaxCrossEntropyLoss
    """
    set_seed(seed)
    if loss_name.upper() == "BCE":
        layers = [
            LinearLayer(input_size=2, output_size=HIDDEN_DIMS[0], enable_bias=True),
            CustomReLU(),
            LinearLayer(input_size=HIDDEN_DIMS[0], output_size=HIDDEN_DIMS[1], enable_bias=True),
            CustomReLU(),
            LinearLayer(input_size=HIDDEN_DIMS[1], output_size=1, enable_bias=True),
            Sigmoid()
        ]
        loss_fn = BinaryCrossEntropyLoss()
    elif loss_name.upper() == "SOFTMAXCE":
        layers = [
            LinearLayer(input_size=2, output_size=HIDDEN_DIMS[0], enable_bias=True),
            CustomReLU(),
            LinearLayer(input_size=HIDDEN_DIMS[0], output_size=HIDDEN_DIMS[1], enable_bias=True),
            CustomReLU(),
            LinearLayer(input_size=HIDDEN_DIMS[1], output_size=2, enable_bias=True)
        ]
        loss_fn = SoftmaxCrossEntropyLoss()
    else:
        raise ValueError(f"Unknown classification loss: {loss_name}")

    optimizer = SGD(lr=CLS_LR, momentum=MOMENTUM)
    return MLP(layers=layers, optimizer=optimizer, loss_fn=loss_fn, seed=seed)

def run_regression_experiments():
    print("\n" + "=" * 70, flush=True)
    print("      RUNNING REGRESSION LOSS EXPERIMENTS (MSE vs MAE)             ", flush=True)
    print(f"      Dataset: cube (1000 samples) | Epochs: {REG_EPOCHS} | Seeds: {SEEDS}     ", flush=True)
    print("=" * 70, flush=True)

    X_train, y_train = load_dataset("regression", "cube", 1000, split="train")
    X_test, y_test = load_dataset("regression", "cube", 1000, split="test")

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_scaled = scaler_X.transform(X_train)
    y_train_scaled = scaler_y.transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)

    loss_types = ["MSE", "MAE"]
    results = {
        loss_type: {
            "test_mse": [],      # (num_seeds, epochs + 1)
            "test_mae": [],      # (num_seeds, epochs + 1)
            "step_magnitude": [] # (num_seeds, epochs)
        }
        for loss_type in loss_types
    }

    for loss_type in loss_types:
        print(f"\n--- Training Regression Model with [{loss_type} Loss] ---", flush=True)
        for seed in SEEDS:
            model = build_regression_model(loss_type, seed=seed)

            prev_params = get_model_params(model)
            seed_test_mse = []
            seed_test_mae = []
            seed_steps = []

            # Evaluate at epoch 0 (untrained initialization)
            def evaluate_test(m: MLP):
                preds_scaled = m.predict(X_test_scaled)
                preds_real = scaler_y.inverse_transform(preds_scaled)
                mse = float(np.mean((y_test - preds_real) ** 2))
                mae = float(np.mean(np.abs(y_test - preds_real)))
                return mse, mae

            init_mse, init_mae = evaluate_test(model)
            seed_test_mse.append(init_mse)
            seed_test_mae.append(init_mae)

            def epoch_callback(epoch: int, m: MLP):
                nonlocal prev_params
                curr_params = get_model_params(m)
                step_mag = compute_step_magnitude(prev_params, curr_params)
                seed_steps.append(step_mag)
                prev_params = curr_params

                mse, mae = evaluate_test(m)
                seed_test_mse.append(mse)
                seed_test_mae.append(mae)

            model.fit(
                X_train_scaled,
                y_train_scaled,
                epochs=REG_EPOCHS,
                batch_size=BATCH_SIZE,
                seed=seed,
                eval_every=1,
                eval_callback=epoch_callback
            )

            results[loss_type]["test_mse"].append(seed_test_mse)
            results[loss_type]["test_mae"].append(seed_test_mae)
            results[loss_type]["step_magnitude"].append(seed_steps)
            print(f"  Seed {seed:3d} finished -> Final Test MSE: {seed_test_mse[-1]:.2f}, Final Test MAE: {seed_test_mae[-1]:.2f}", flush=True)

    # Convert results to numpy arrays
    for loss_type in loss_types:
        for k in results[loss_type]:
            results[loss_type][k] = np.array(results[loss_type][k])

    # -------------------------------------------------------------
    # Plot 1: Regression Test Metrics (2 subplots: MSE model vs MAE model)
    # -------------------------------------------------------------
    print("\nGenerating Figure 1: Regression Test Metrics (MSE vs MAE)...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Regression Model Performance on Test Set: MSE Loss vs MAE Loss (cube 1000)", fontsize=14, fontweight="bold", y=0.98)

    epochs_axis = np.arange(0, REG_EPOCHS + 1)

    for idx, loss_type in enumerate(loss_types):
        ax_mse = axes[idx]
        ax_mae = ax_mse.twinx()

        mse_data = results[loss_type]["test_mse"]
        mae_data = results[loss_type]["test_mae"]

        mse_mean = np.mean(mse_data, axis=0)
        mae_mean = np.mean(mae_data, axis=0)

        # Left axis: Test MSE (Blue)
        line1 = ax_mse.plot(epochs_axis, mse_mean, color="#1f77b4", linewidth=2.4, marker="o", markersize=4, markevery=4, label="Test MSE (left axis)")
        ax_mse.set_ylabel("Test MSE", color="#1f77b4", fontweight="bold", fontsize=11)
        ax_mse.tick_params(axis="y", labelcolor="#1f77b4")
        ax_mse.grid(True, linestyle="--", alpha=0.5)

        # Right axis: Test MAE (Orange)
        line2 = ax_mae.plot(epochs_axis, mae_mean, color="#d95f02", linewidth=2.4, linestyle="--", marker="s", markersize=4, markevery=4, label="Test MAE (right axis)")
        ax_mae.set_ylabel("Test MAE", color="#d95f02", fontweight="bold", fontsize=11)
        ax_mae.tick_params(axis="y", labelcolor="#d95f02")

        ax_mse.set_xlabel("Epoch", fontweight="bold", fontsize=11)
        ax_mse.set_title(f"Model Trained with {loss_type} Loss", fontsize=12, fontweight="bold")

        # Unified Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_mse.legend(lines, labels, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig1_path = RESULTS_DIR / "regression_test_metrics.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig1_path}", flush=True)

    # -------------------------------------------------------------
    # Plot 2: Regression Weight Update Step Magnitude (MSE vs MAE)
    # -------------------------------------------------------------
    print("Generating Figure 2: Regression Weight Update Step Magnitude...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Regression: Weight Update Step Magnitude across Epochs (cube 1000)", fontsize=14, fontweight="bold", y=0.98)

    step_epochs_axis = np.arange(1, REG_EPOCHS + 1)

    colors = {"MSE": "#1f77b4", "MAE": "#d95f02"}

    for idx, loss_type in enumerate(loss_types):
        ax = axes[idx]
        step_data = results[loss_type]["step_magnitude"]
        step_mean = np.mean(step_data, axis=0)

        ax.plot(
            step_epochs_axis,
            step_mean,
            color=colors[loss_type],
            linewidth=2.4,
            marker="o",
            markersize=4,
            markevery=3,
            label=f"{loss_type} Update Step"
        )

        ax.set_title(f"Trained with {loss_type} Loss", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontweight="bold", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.9)

        if idx == 0:
            ax.set_ylabel("Weight Step Magnitude ||W(t) - W(t-1)||", fontweight="bold", fontsize=11)

    plt.tight_layout()
    fig2_path = RESULTS_DIR / "regression_weight_updates.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig2_path}", flush=True)

def run_classification_experiments():
    print("\n" + "=" * 70, flush=True)
    print("      RUNNING CLASSIFICATION LOSS EXPERIMENTS (BCE vs SoftmaxCE)  ", flush=True)
    print(f"      Dataset: simple (1000 samples) | Epochs: {CLS_EPOCHS} | Seeds: {SEEDS} ", flush=True)
    print("=" * 70, flush=True)

    X_train, y_train = load_dataset("classification", "simple", 1000, split="train")
    X_test, y_test = load_dataset("classification", "simple", 1000, split="test")

    scaler_X = StandardScaler().fit(X_train)
    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Targets for BCE: binary scalar 0.0 or 1.0 (class 1 -> 0, class 2 -> 1)
    y_train_bce = (y_train == 2).astype(float).reshape(-1, 1)
    y_test_bce = (y_test == 2).astype(int).reshape(-1, 1)

    # Targets for SoftmaxCE: 2-class one-hot vectors
    y_train_oh = to_one_hot(y_train, num_classes=2)
    y_test_labels = (y_test - 1).astype(int)

    loss_types = ["BCE", "SoftmaxCE"]
    results = {
        loss_type: {
            "test_accuracy": [], # (num_seeds, epochs + 1)
            "step_magnitude": [] # (num_seeds, epochs)
        }
        for loss_type in loss_types
    }

    for loss_type in loss_types:
        print(f"\n--- Training Classification Model with [{loss_type} Loss] ---", flush=True)
        for seed in SEEDS:
            model = build_classification_model(loss_type, seed=seed)

            prev_params = get_model_params(model)
            seed_test_acc = []
            seed_steps = []

            # Evaluate at epoch 0 (untrained initialization)
            def evaluate_acc(m: MLP) -> float:
                preds = m.predict(X_test_scaled)
                if loss_type == "BCE":
                    pred_cls = (preds >= 0.5).astype(int)
                    acc = float(np.mean(pred_cls == y_test_bce) * 100.0)
                else:
                    pred_cls = np.argmax(preds, axis=1)
                    acc = float(np.mean(pred_cls == y_test_labels) * 100.0)
                return acc

            init_acc = evaluate_acc(model)
            seed_test_acc.append(init_acc)

            def epoch_callback(epoch: int, m: MLP):
                nonlocal prev_params
                curr_params = get_model_params(m)
                step_mag = compute_step_magnitude(prev_params, curr_params)
                seed_steps.append(step_mag)
                prev_params = curr_params

                acc = evaluate_acc(m)
                seed_test_acc.append(acc)

            y_tr = y_train_bce if loss_type == "BCE" else y_train_oh

            model.fit(
                X_train_scaled,
                y_tr,
                epochs=CLS_EPOCHS,
                batch_size=BATCH_SIZE,
                seed=seed,
                eval_every=1,
                eval_callback=epoch_callback
            )

            results[loss_type]["test_accuracy"].append(seed_test_acc)
            results[loss_type]["step_magnitude"].append(seed_steps)
            print(f"  Seed {seed:3d} finished -> Final Test Accuracy: {seed_test_acc[-1]:.2f}%", flush=True)

    # Convert results to numpy arrays
    for loss_type in loss_types:
        for k in results[loss_type]:
            results[loss_type][k] = np.array(results[loss_type][k])

    # -------------------------------------------------------------
    # Plot 3: Classification Test Accuracy (2 subplots: BCE vs SoftmaxCE)
    # -------------------------------------------------------------
    print("\nGenerating Figure 3: Classification Test Accuracy (BCE vs SoftmaxCE)...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Classification Model Test Accuracy: Binary Cross-Entropy vs Softmax Cross-Entropy (simple 1000)", fontsize=14, fontweight="bold", y=0.98)

    epochs_axis = np.arange(0, CLS_EPOCHS + 1)
    colors = {"BCE": "#2ca02c", "SoftmaxCE": "#9467bd"}

    for idx, loss_type in enumerate(loss_types):
        ax = axes[idx]
        acc_data = results[loss_type]["test_accuracy"]
        acc_mean = np.mean(acc_data, axis=0)

        display_name = "Binary Cross-Entropy" if loss_type == "BCE" else "Softmax Cross-Entropy"

        ax.plot(
            epochs_axis,
            acc_mean,
            color=colors[loss_type],
            linewidth=2.4,
            marker="o",
            markersize=5,
            markevery=2,
            label=f"{loss_type} Test Accuracy"
        )

        ax.set_title(f"Trained with {display_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontweight="bold", fontsize=11)
        ax.set_ylim([45, 102])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", framealpha=0.9)

        if idx == 0:
            ax.set_ylabel("Test Accuracy (%)", fontweight="bold", fontsize=11)

    plt.tight_layout()
    fig3_path = RESULTS_DIR / "classification_test_accuracy.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig3_path}", flush=True)

    # -------------------------------------------------------------
    # Plot 4: Classification Weight Update Step Magnitude (BCE vs SoftmaxCE)
    # -------------------------------------------------------------
    print("Generating Figure 4: Classification Weight Update Step Magnitude...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Classification: Weight Update Step Magnitude across Epochs (simple 1000)", fontsize=14, fontweight="bold", y=0.98)

    step_epochs_axis = np.arange(1, CLS_EPOCHS + 1)

    for idx, loss_type in enumerate(loss_types):
        ax = axes[idx]
        step_data = results[loss_type]["step_magnitude"]
        step_mean = np.mean(step_data, axis=0)

        display_name = "Binary Cross-Entropy" if loss_type == "BCE" else "Softmax Cross-Entropy"

        ax.plot(
            step_epochs_axis,
            step_mean,
            color=colors[loss_type],
            linewidth=2.4,
            marker="o",
            markersize=5,
            markevery=2,
            label=f"{loss_type} Update Step"
        )

        ax.set_title(f"Trained with {display_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontweight="bold", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.9)

        if idx == 0:
            ax.set_ylabel("Weight Step Magnitude ||W(t) - W(t-1)||", fontweight="bold", fontsize=11)

    plt.tight_layout()
    fig4_path = RESULTS_DIR / "classification_weight_updates.png"
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig4_path}", flush=True)

if __name__ == "__main__":
    run_regression_experiments()
    run_classification_experiments()
    print("\n" + "=" * 70)
    print("All Loss Function Experiments Successfully Completed!")
    print(f"Figures saved in: {RESULTS_DIR}")
    print("=" * 70 + "\n")
