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
from src.core.activations import CustomReLU, Linear
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

# Hyperparameters
REG_EPOCHS = 75
REG_EVAL_EVERY = 3

CLS_EPOCHS = 40
CLS_EVAL_EVERY = 2

SEEDS = [42, 123, 999]
SAMPLE_SIZES = [100, 1000, 10000]

# Network architecture patterns
PATTERNS = {
    "Small Layers": {
        0: [],
        1: [16],
        2: [16, 16],
        3: [16, 16, 16],
        4: [16, 16, 16, 16],
    },
    "Big Layers": {
        0: [],
        1: [64],
        2: [64, 64],
        3: [64, 64, 64],
        4: [64, 64, 64, 64],
    },
    "Pyramid": {
        0: [],
        1: [32],
        2: [16, 32],
        3: [16, 64, 16],
        4: [16, 64, 64, 16],
    },
    "Reversed Pyramid": {
        0: [],
        1: [32],
        2: [64, 32],
        3: [64, 16, 64],
        4: [64, 16, 16, 64],
    },
}

DEPTH_STYLES = {
    0: {"color": "#444444", "linestyle": "--", "marker": "s", "label_suffix": "0 Layers (Linear Baseline)"},
    1: {"color": "#1f77b4", "linestyle": "-",  "marker": "o", "label_suffix": "1 Layer"},
    2: {"color": "#2ca02c", "linestyle": "-",  "marker": "^", "label_suffix": "2 Layers"},
    3: {"color": "#ff7f0e", "linestyle": "-",  "marker": "D", "label_suffix": "3 Layers"},
    4: {"color": "#d62728", "linestyle": "-",  "marker": "v", "label_suffix": "4 Layers"},
}

def build_regression_mlp(hidden_dims: list[int], seed: int, lr: float = 0.002) -> MLP:
    """
    Builds an MLP for 1D regression with given hidden layer dimensions.
    Uses lr=0.002 with momentum=0.9 for numerical stability across deep (4-layer) networks.
    """
    set_seed(seed)
    layers = []
    in_dim = 1
    for h_dim in hidden_dims:
        layers.append(LinearLayer(input_size=in_dim, output_size=h_dim, enable_bias=True))
        layers.append(CustomReLU())
        in_dim = h_dim
    # Output layer
    layers.append(LinearLayer(input_size=in_dim, output_size=1, enable_bias=True))
    layers.append(Linear())
    optimizer = SGD(lr=lr, momentum=0.9)
    return MLP(layers=layers, optimizer=optimizer, loss_fn=MSELoss(), seed=seed)

def build_classification_mlp(hidden_dims: list[int], seed: int, lr: float = 0.01) -> MLP:
    """
    Builds an MLP for 3-class classification with given hidden layer dimensions.
    Uses lr=0.01 with momentum=0.9.
    """
    set_seed(seed)
    layers = []
    in_dim = 2
    for h_dim in hidden_dims:
        layers.append(LinearLayer(input_size=in_dim, output_size=h_dim, enable_bias=True))
        layers.append(CustomReLU())
        in_dim = h_dim
    # Output layer (logits for SoftmaxCrossEntropyLoss)
    layers.append(LinearLayer(input_size=in_dim, output_size=3, enable_bias=True))
    optimizer = SGD(lr=lr, momentum=0.9)
    return MLP(layers=layers, optimizer=optimizer, loss_fn=SoftmaxCrossEntropyLoss(), seed=seed)

def run_regression_experiments():
    print("\n====================================================================", flush=True)
    print("      RUNNING REGRESSION EXPERIMENTS (data.cube)                   ", flush=True)
    print(f"      ({REG_EPOCHS} Epochs, evaluated every {REG_EVAL_EVERY} epochs, Seeds: {SEEDS})", flush=True)
    print("====================================================================", flush=True)

    checkpoint_epochs = list(range(0, REG_EPOCHS + 1, REG_EVAL_EVERY))
    results = {
        pattern: {size: {} for size in SAMPLE_SIZES}
        for pattern in PATTERNS
    }

    for size in SAMPLE_SIZES:
        print(f"\n=======================================================", flush=True)
        print(f"--- Loading Regression Dataset (cube, {size} samples) ---", flush=True)
        print(f"=======================================================", flush=True)
        X_train, y_train = load_dataset("regression", "cube", size, split="train")
        X_test, y_test = load_dataset("regression", "cube", size, split="test")

        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train)

        X_train_scaled = scaler_X.transform(X_train)
        y_train_scaled = scaler_y.transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)

        batch_size = 16 if size == 100 else (32 if size == 1000 else 64)

        # Cache architectures across patterns (e.g., depth 0 is identical for all 4 patterns)
        arch_cache = {}

        for pattern_name, depth_dict in PATTERNS.items():
            print(f"\n--- Pattern: [{pattern_name}] (Dataset size: {size}) ---", flush=True)
            for depth in range(5):
                arch_tuple = tuple(depth_dict[depth])
                
                if arch_tuple in arch_cache:
                    results[pattern_name][size][depth] = arch_cache[arch_tuple]
                    final_m = np.mean(arch_cache[arch_tuple][:, -1])
                    print(f"  [Depth {depth}]: {list(arch_tuple)} (reused cached metrics -> Final MSE: {final_m:.4f})", flush=True)
                    continue

                print(f"  [Depth {depth}]: {list(arch_tuple)} -> Training over seeds {SEEDS}...", flush=True)
                seed_metrics = []

                for seed in SEEDS:
                    model = build_regression_mlp(depth_dict[depth], seed=seed, lr=0.002)
                    checkpoints = []

                    def eval_step(epoch, m):
                        preds_scaled = m.predict(X_test_scaled)
                        if np.isnan(preds_scaled).any() or np.isinf(preds_scaled).any():
                            test_mse = 1e5
                        else:
                            preds_real = scaler_y.inverse_transform(preds_scaled)
                            test_mse = float(np.mean((y_test - preds_real) ** 2))
                        checkpoints.append(test_mse)

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

                metrics_array = np.array(seed_metrics)
                final_mean = np.mean(metrics_array[:, -1])
                print(f"    -> Finished {len(SEEDS)} seeds. Final Test MSE (mean): {final_mean:.4f}", flush=True)

                arch_cache[arch_tuple] = metrics_array
                results[pattern_name][size][depth] = metrics_array

    # Plot 4x3 Grid for Regression
    print("\nGenerating Regression Summary Figure (4x3 Grid)...", flush=True)
    fig, axes = plt.subplots(4, 3, figsize=(22, 20), sharex=True)

    pattern_names = list(PATTERNS.keys())
    for row_idx, pattern_name in enumerate(pattern_names):
        for col_idx, size in enumerate(SAMPLE_SIZES):
            ax = axes[row_idx, col_idx]

            for depth in range(5):
                cfg = DEPTH_STYLES[depth]
                matrix = results[pattern_name][size][depth]
                mean_curve = np.mean(matrix, axis=0)

                if depth == 0:
                    label = cfg["label_suffix"]
                else:
                    label = f"{cfg['label_suffix']}: {PATTERNS[pattern_name][depth]}"

                ax.plot(
                    checkpoint_epochs,
                    mean_curve,
                    label=label,
                    color=cfg["color"],
                    linestyle=cfg["linestyle"],
                    linewidth=2.2,
                    marker=cfg["marker"],
                    markersize=4,
                    markevery=3,
                    alpha=0.9
                )

            # Set log scale on Y-axis for clear separation of depths
            ax.set_yscale("log")

            # Titles and Labels
            if row_idx == 0:
                ax.set_title(f"Dataset Size: {size} samples", fontsize=14, fontweight="bold", pad=12)
            if col_idx == 0:
                ax.set_ylabel(f"{pattern_name}\n\nTest MSE (Log Scale)", fontsize=12, fontweight="bold")
            else:
                ax.set_ylabel("Test MSE (Log Scale)", fontsize=11)

            if row_idx == len(pattern_names) - 1:
                ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")

            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Influence of Network Depth and Width on Regression (data.cube)\nEvaluated on Test MSE Across Epochs (Mean of 3 Seeds, Log Scale)",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )
    plt.tight_layout()
    save_path = RESULTS_DIR / "regression_depth_width_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] Regression comparison plot -> {save_path}", flush=True)

def run_classification_experiments():
    print("\n====================================================================", flush=True)
    print("    RUNNING CLASSIFICATION EXPERIMENTS (data.three_gauss)          ", flush=True)
    print(f"      ({CLS_EPOCHS} Epochs, evaluated every {CLS_EVAL_EVERY} epochs, Seeds: {SEEDS})", flush=True)
    print("====================================================================", flush=True)

    checkpoint_epochs = list(range(0, CLS_EPOCHS + 1, CLS_EVAL_EVERY))
    results = {
        pattern: {size: {} for size in SAMPLE_SIZES}
        for pattern in PATTERNS
    }

    for size in SAMPLE_SIZES:
        print(f"\n=======================================================", flush=True)
        print(f"--- Loading Classification Dataset (three_gauss, {size} samples) ---", flush=True)
        print(f"=======================================================", flush=True)
        X_train, y_train = load_dataset("classification", "three_gauss", size, split="train")
        X_test, y_test = load_dataset("classification", "three_gauss", size, split="test")

        print_class_distribution(y_test, dataset_name=f"three_gauss.test.{size}")

        scaler_X = StandardScaler().fit(X_train)
        X_train_scaled = scaler_X.transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_oh = to_one_hot(y_train, num_classes=3)

        batch_size = 16 if size == 100 else (32 if size == 1000 else 64)

        arch_cache = {}

        for pattern_name, depth_dict in PATTERNS.items():
            print(f"\n--- Pattern: [{pattern_name}] (Dataset size: {size}) ---", flush=True)
            for depth in range(5):
                arch_tuple = tuple(depth_dict[depth])

                if arch_tuple in arch_cache:
                    results[pattern_name][size][depth] = arch_cache[arch_tuple]
                    final_m = np.mean(arch_cache[arch_tuple][:, -1])
                    print(f"  [Depth {depth}]: {list(arch_tuple)} (reused cached metrics -> Final Acc: {final_m:.2f}%)", flush=True)
                    continue

                print(f"  [Depth {depth}]: {list(arch_tuple)} -> Training over seeds {SEEDS}...", flush=True)
                seed_metrics = []

                for seed in SEEDS:
                    model = build_classification_mlp(depth_dict[depth], seed=seed, lr=0.01)
                    checkpoints = []

                    def eval_step(epoch, m):
                        logits = m.predict(X_test_scaled)
                        if np.isnan(logits).any() or np.isinf(logits).any():
                            acc = 0.0
                        else:
                            pred_classes = np.argmax(logits, axis=1) + 1  # 1-indexed classes
                            acc = float(np.mean(pred_classes == y_test) * 100.0)
                        checkpoints.append(acc)

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

                metrics_array = np.array(seed_metrics)
                final_mean = np.mean(metrics_array[:, -1])
                print(f"    -> Finished {len(SEEDS)} seeds. Final Test Accuracy (mean): {final_mean:.2f}%", flush=True)

                arch_cache[arch_tuple] = metrics_array
                results[pattern_name][size][depth] = metrics_array

    # Plot 4x3 Grid for Classification
    print("\nGenerating Classification Summary Figure (4x3 Grid)...", flush=True)
    fig, axes = plt.subplots(4, 3, figsize=(22, 20), sharex=True, sharey=True)

    pattern_names = list(PATTERNS.keys())
    for row_idx, pattern_name in enumerate(pattern_names):
        for col_idx, size in enumerate(SAMPLE_SIZES):
            ax = axes[row_idx, col_idx]

            for depth in range(5):
                cfg = DEPTH_STYLES[depth]
                matrix = results[pattern_name][size][depth]
                mean_curve = np.mean(matrix, axis=0)

                if depth == 0:
                    label = cfg["label_suffix"]
                else:
                    label = f"{cfg['label_suffix']}: {PATTERNS[pattern_name][depth]}"

                ax.plot(
                    checkpoint_epochs,
                    mean_curve,
                    label=label,
                    color=cfg["color"],
                    linestyle=cfg["linestyle"],
                    linewidth=2.2,
                    marker=cfg["marker"],
                    markersize=4,
                    markevery=2,
                    alpha=0.9
                )

            if row_idx == 0:
                ax.set_title(f"Dataset Size: {size} samples", fontsize=14, fontweight="bold", pad=12)
            if col_idx == 0:
                ax.set_ylabel(f"{pattern_name}\n\nTest Accuracy (%)", fontsize=12, fontweight="bold")
            else:
                ax.set_ylabel("Test Accuracy (%)", fontsize=11)

            if row_idx == len(pattern_names) - 1:
                ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")

            ax.set_ylim([30, 102])
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    fig.suptitle(
        "Influence of Network Depth and Width on Classification (data.three_gauss)\nEvaluated on Test Accuracy (%) Across Epochs (Mean of 3 Seeds)",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )
    plt.tight_layout()
    save_path = RESULTS_DIR / "classification_depth_width_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] Classification comparison plot -> {save_path}", flush=True)

if __name__ == "__main__":
    print("====================================================================", flush=True)
    print(" Starting Experiment 02: Network Depth and Layer Width Analysis     ", flush=True)
    print("====================================================================", flush=True)
    run_regression_experiments()
    run_classification_experiments()
    print("\n====================================================================", flush=True)
    print(" Experiment 02 Completed Successfully! All figures saved in results/", flush=True)
    print("====================================================================", flush=True)
