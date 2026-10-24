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
from src.core.activations import CustomReLU, Sigmoid, Tanh, Linear
from src.core.loss import MSELoss, SoftmaxCrossEntropyLoss
from src.core.optimizer import SGD
from src.core.network import MLP, set_seed
from src.utils.data_loader import (
    load_dataset,
    print_class_distribution,
    to_one_hot,
    StandardScaler
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment hyperparameters
REG_EPOCHS = 50
REG_EVAL_EVERY = 2

CLS_EPOCHS = 20
CLS_EVAL_EVERY = 1

SEEDS = [42, 123, 999]
SAMPLE_SIZES = [100, 1000, 10000]

ACTIVATIONS = [
    ("ReLU", CustomReLU, "royalblue"),
    ("Sigmoid", Sigmoid, "forestgreen"),
    ("Tanh", Tanh, "crimson")
]

def build_regression_mlp(activation_cls, seed: int) -> MLP:
    """Builds a standardized 2-hidden layer MLP for regression."""
    set_seed(seed)
    layers = [
        LinearLayer(input_size=1, output_size=32, enable_bias=True),
        activation_cls(),
        LinearLayer(input_size=32, output_size=16, enable_bias=True),
        activation_cls(),
        LinearLayer(input_size=16, output_size=1, enable_bias=True),
        Linear()
    ]
    optimizer = SGD(lr=0.01, momentum=0.9)
    return MLP(layers=layers, optimizer=optimizer, loss_fn=MSELoss(), seed=seed)

def build_classification_mlp(activation_cls, seed: int) -> MLP:
    """Builds a standardized 2-hidden layer MLP for 3-class classification."""
    set_seed(seed)
    layers = [
        LinearLayer(input_size=2, output_size=32, enable_bias=True),
        activation_cls(),
        LinearLayer(input_size=32, output_size=16, enable_bias=True),
        activation_cls(),
        LinearLayer(input_size=16, output_size=3, enable_bias=True)
    ]
    optimizer = SGD(lr=0.02, momentum=0.9)
    return MLP(layers=layers, optimizer=optimizer, loss_fn=SoftmaxCrossEntropyLoss(), seed=seed)

def run_regression_experiments():
    print("\n========================================================")
    print("      RUNNING REGRESSION EXPERIMENTS (data.cube)        ")
    print(f"     ({REG_EPOCHS} Epochs, evaluated & printed every {REG_EVAL_EVERY} epochs)    ")
    print("========================================================")

    checkpoint_epochs = list(range(0, REG_EPOCHS + 1, REG_EVAL_EVERY))
    results = {size: {act_name: [] for act_name, _, _ in ACTIVATIONS} for size in SAMPLE_SIZES}
    trained_models_1000 = {}
    scalers_1000 = {}

    for size in SAMPLE_SIZES:
        print(f"\n--- Loading Regression Dataset (cube, {size} samples) ---")
        X_train, y_train = load_dataset("regression", "cube", size, split="train")
        X_test, y_test = load_dataset("regression", "cube", size, split="test")

        # Standardize features and targets using train set parameters
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train)

        X_train_scaled = scaler_X.transform(X_train)
        y_train_scaled = scaler_y.transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)

        if size == 1000:
            scalers_1000["X"] = scaler_X
            scalers_1000["y"] = scaler_y
            scalers_1000["test_data"] = (X_test, y_test)

        batch_size = 16 if size == 100 else (32 if size == 1000 else 64)

        for act_name, act_cls, _ in ACTIVATIONS:
            print(f"\nTraining [{act_name}] on {size} samples (evaluating every {REG_EVAL_EVERY} epochs)...")
            seed_metrics = []

            for seed in SEEDS:
                model = build_regression_mlp(act_cls, seed)
                checkpoints = []

                def eval_step(epoch, m):
                    # Predict on scaled test set and convert back to original scale
                    preds_scaled = m.predict(X_test_scaled)
                    preds_real = scaler_y.inverse_transform(preds_scaled)
                    test_mse = np.mean((y_test - preds_real) ** 2)
                    checkpoints.append(test_mse)
                    print(f"  [Size {size:5d} | {act_name:7s} | Seed {seed}] Epoch {epoch:2d}: Test MSE = {test_mse:.2f}")

                # Record epoch 0 before training starts
                eval_step(0, model)

                model.fit(
                    X_train_scaled,
                    y_train_scaled,
                    epochs=REG_EPOCHS,
                    batch_size=batch_size,
                    seed=seed,
                    eval_every=REG_EVAL_EVERY,
                    eval_callback=eval_step
                )
                seed_metrics.append(checkpoints)

                if size == 1000 and seed == SEEDS[0]:
                    trained_models_1000[act_name] = model

            results[size][act_name] = np.array(seed_metrics)

    # Plot 1: 1x3 Subplots of Regression Convergence (Test MSE vs Epoch)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for idx, size in enumerate(SAMPLE_SIZES):
        ax = axes[idx]
        for act_name, _, color in ACTIVATIONS:
            matrix = results[size][act_name]  # (num_seeds, num_checkpoints)
            mean_vals = np.mean(matrix, axis=0)

            ax.plot(checkpoint_epochs, mean_vals, label=act_name, color=color, linewidth=2, marker='o', markersize=3)

        ax.set_title(f"Dataset Size: {size} samples", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Test MSE", fontsize=11)
        ax.set_xticks(range(0, REG_EPOCHS + 1, 5))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10)

    fig.suptitle("Regression Convergence Speed by Activation Function (data.cube)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = RESULTS_DIR / "regression_activation_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] Regression comparison plot -> {save_path}")

    # Plot 2: Fitted curves on test points (1000 samples)
    if trained_models_1000:
        X_test_1000, y_test_1000 = scalers_1000["test_data"]
        scaler_X = scalers_1000["X"]
        scaler_y = scalers_1000["y"]

        plt.figure(figsize=(10, 6))
        # Plot subset of test points for clean visualization
        sort_idx = np.argsort(X_test_1000.ravel())
        x_dense = np.linspace(X_test_1000.min(), X_test_1000.max(), 500).reshape(-1, 1)
        x_dense_scaled = scaler_X.transform(x_dense)

        plt.scatter(X_test_1000[sort_idx][::5], y_test_1000[sort_idx][::5], color="gray", alpha=0.5, s=20, label="Test Data")

        for act_name, _, color in ACTIVATIONS:
            model = trained_models_1000[act_name]
            preds_scaled = model.predict(x_dense_scaled)
            preds_real = scaler_y.inverse_transform(preds_scaled)
            plt.plot(x_dense.ravel(), preds_real.ravel(), label=f"{act_name} Fit", color=color, linewidth=2.5)

        plt.title("Regression Function Approximation (1000 samples, data.cube)", fontsize=13, fontweight="bold")
        plt.xlabel("X", fontsize=11)
        plt.ylabel("Y", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()
        fit_path = RESULTS_DIR / "regression_fit_comparison.png"
        plt.savefig(fit_path, dpi=300)
        plt.close()
        print(f"[Saved] Regression fit plot -> {fit_path}")

def run_classification_experiments():
    print("\n========================================================")
    print("   RUNNING CLASSIFICATION EXPERIMENTS (data.three_gauss) ")
    print(f"     ({CLS_EPOCHS} Epochs, evaluated & printed every {CLS_EVAL_EVERY} epoch)    ")
    print("========================================================")

    checkpoint_epochs = list(range(0, CLS_EPOCHS + 1, CLS_EVAL_EVERY))
    results = {size: {act_name: [] for act_name, _, _ in ACTIVATIONS} for size in SAMPLE_SIZES}
    trained_models_1000 = {}
    test_data_1000 = {}

    for size in SAMPLE_SIZES:
        print(f"\n--- Loading Classification Dataset (three_gauss, {size} samples) ---")
        X_train, y_train = load_dataset("classification", "three_gauss", size, split="train")
        X_test, y_test = load_dataset("classification", "three_gauss", size, split="test")

        # Requirement 1: Make a count of each class in test set
        print_class_distribution(y_test, dataset_name=f"three_gauss.test.{size}")

        # Standardize features
        scaler_X = StandardScaler().fit(X_train)
        X_train_scaled = scaler_X.transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # One-hot encode targets for SoftmaxCrossEntropyLoss (3 classes: 1, 2, 3)
        y_train_oh = to_one_hot(y_train, num_classes=3)

        if size == 1000:
            test_data_1000["X"] = X_test_scaled
            test_data_1000["y"] = y_test
            test_data_1000["scaler"] = scaler_X

        batch_size = 16 if size == 100 else (32 if size == 1000 else 64)

        for act_name, act_cls, _ in ACTIVATIONS:
            print(f"\nTraining [{act_name}] on {size} samples (evaluating every {CLS_EVAL_EVERY} epoch)...")
            seed_metrics = []

            for seed in SEEDS:
                model = build_classification_mlp(act_cls, seed)
                checkpoints = []

                def eval_step(epoch, m):
                    # Compute classification accuracy (%)
                    logits = m.predict(X_test_scaled)
                    pred_classes = np.argmax(logits, axis=1) + 1  # 1-indexed classes
                    acc = np.mean(pred_classes == y_test) * 100.0
                    checkpoints.append(acc)
                    print(f"  [Size {size:5d} | {act_name:7s} | Seed {seed}] Epoch {epoch:2d}: Test Accuracy = {acc:.2f}%")

                # Record epoch 0 before training starts
                eval_step(0, model)

                model.fit(
                    X_train_scaled,
                    y_train_oh,
                    epochs=CLS_EPOCHS,
                    batch_size=batch_size,
                    seed=seed,
                    eval_every=CLS_EVAL_EVERY,
                    eval_callback=eval_step
                )
                seed_metrics.append(checkpoints)

                if size == 1000 and seed == SEEDS[0]:
                    trained_models_1000[act_name] = model

            results[size][act_name] = np.array(seed_metrics)

    # Plot 1: 1x3 Subplots of Classification Accuracy vs Epoch
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, size in enumerate(SAMPLE_SIZES):
        ax = axes[idx]
        for act_name, _, color in ACTIVATIONS:
            matrix = results[size][act_name]
            mean_vals = np.mean(matrix, axis=0)

            ax.plot(checkpoint_epochs, mean_vals, label=act_name, color=color, linewidth=2, marker='o', markersize=3)

        ax.set_title(f"Dataset Size: {size} samples", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Test Accuracy (%)", fontsize=11)
        ax.set_xticks(range(0, CLS_EPOCHS + 1, 2))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10, loc="lower right")
        ax.set_ylim([25, 100])

    fig.suptitle("Classification Convergence Speed by Activation Function (data.three_gauss)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = RESULTS_DIR / "classification_activation_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] Classification comparison plot -> {save_path}")

    # Plot 2: 2D Decision Boundaries for the 3 activations (1000 samples)
    if trained_models_1000:
        X_test_scaled = test_data_1000["X"]
        y_test = test_data_1000["y"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        x_min, x_max = X_test_scaled[:, 0].min() - 0.5, X_test_scaled[:, 0].max() + 0.5
        y_min, y_max = X_test_scaled[:, 1].min() - 0.5, X_test_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for idx, (act_name, _, _) in enumerate(ACTIVATIONS):
            ax = axes[idx]
            model = trained_models_1000[act_name]
            logits = model.predict(grid)
            zz = (np.argmax(logits, axis=1) + 1).reshape(xx.shape)

            ax.contourf(xx, yy, zz, alpha=0.25, cmap="Set1")
            ax.contour(xx, yy, zz, colors="black", linewidths=0.7, alpha=0.7)

            for c_idx, cls_val in enumerate(np.unique(y_test)):
                mask = (y_test == cls_val)
                ax.scatter(
                    X_test_scaled[mask, 0],
                    X_test_scaled[mask, 1],
                    color=colors[c_idx],
                    edgecolors="k",
                    s=25,
                    label=f"Class {cls_val}",
                    alpha=0.8
                )

            ax.set_title(f"Decision Boundary: {act_name}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Scaled Feature 1", fontsize=10)
            ax.set_ylabel("Scaled Feature 2", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)
            if idx == 0:
                ax.legend(fontsize=9, loc="upper right")

        fig.suptitle("Decision Boundaries Comparison (1000 samples, data.three_gauss)", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        bound_path = RESULTS_DIR / "classification_decision_boundaries.png"
        plt.savefig(bound_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Saved] Decision boundaries plot -> {bound_path}")

if __name__ == "__main__":
    run_regression_experiments()
    run_classification_experiments()
    print("\nAll activation function experiments finished successfully!")
