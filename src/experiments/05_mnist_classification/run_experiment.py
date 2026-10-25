import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.layer import LinearLayer
from src.core.activations import CustomReLU
from src.core.loss import SoftmaxCrossEntropyLoss
from src.core.optimizer import SGD
from src.core.network import MLP, set_seed
from src.utils.data_loader import load_mnist, to_one_hot
from src.visualization.visualizer import NetworkVisualizer

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 0.03
MOMENTUM = 0.9
SEED = 42
HIDDEN_DIMS = [256, 128, 64]

# Matplotlib styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Computes confusion matrix where rows are true classes and columns are predicted classes."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def plot_confusion_matrix_percentage(cm: np.ndarray, accuracy_pct: float, save_path: Path):
    """Plots and saves the confusion matrix with normalized percentages per class."""
    visualizer = NetworkVisualizer()
    class_names = [f"Digit {i}" for i in range(10)]

    title_str = f"MNIST Test Set Confusion Matrix (Accuracy={accuracy_pct:.2f}%)"
    visualizer.visualise_confusion_matrix(
        cm=cm,
        class_names=class_names,
        title=title_str,
        cmap="Blues",
        normalize=True,
        save_path=str(save_path),
        show=False
    )
    print(f"Saved confusion matrix (with percentage) to: {save_path}")


def run_mnist_experiment():
    print("=" * 75)
    print("  EXPERIMENT 5: MNIST HANDWRITTEN DIGITS CLASSIFICATION WITH MLP  ")
    print(f"  Architecture: 784 -> {' -> '.join(str(d) for d in HIDDEN_DIMS)} -> 10 (SoftmaxCE) ")
    print(f"  Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE} | Seed: {SEED} ")
    print("=" * 75)

    # 1. Load MNIST data
    print("\nLoading MNIST dataset...")
    X_train, y_train = load_mnist(split="train", normalize=True, flatten=True)
    X_test, y_test = load_mnist(split="test", normalize=True, flatten=True)

    print(f"  Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")

    y_train_oh = to_one_hot(y_train, num_classes=10)

    # 2. Build Model
    print("Building Multi-Layer Perceptron (MLP)...")
    set_seed(SEED)
    layers = [
        LinearLayer(input_size=784, output_size=HIDDEN_DIMS[0], enable_bias=True),
        CustomReLU(),
        LinearLayer(input_size=HIDDEN_DIMS[0], output_size=HIDDEN_DIMS[1], enable_bias=True),
        CustomReLU(),
        LinearLayer(input_size=HIDDEN_DIMS[1], output_size=HIDDEN_DIMS[2], enable_bias=True),
        CustomReLU(),
        LinearLayer(input_size=HIDDEN_DIMS[2], output_size=10, enable_bias=True)
    ]
    optimizer = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_fn = SoftmaxCrossEntropyLoss()
    model = MLP(layers=layers, optimizer=optimizer, loss_fn=loss_fn, seed=SEED)

    # 3. Train the model
    print(f"Training MLP for {EPOCHS} epochs (batch size: {BATCH_SIZE})...")
    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train_oh,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    elapsed = time.perf_counter() - start_time
    print(f"Training completed in {elapsed:.2f} seconds ({elapsed / EPOCHS:.2f} s/epoch).")

    # 4. Evaluate on Test Set & Generate Confusion Matrix (with percentage)
    print("\nEvaluating on Test Set...")
    test_logits = model.predict(X_test)
    test_preds = np.argmax(test_logits, axis=1)

    # Overall general test accuracy percentage
    test_acc_pct = float(np.mean(test_preds == y_test)) * 100.0
    print(f"General Test Accuracy: {test_acc_pct:.2f}%")

    # Confusion Matrix
    cm = compute_confusion_matrix(y_test, test_preds, num_classes=10)

    # Save only the confusion matrix with percentage
    cm_plot_path = RESULTS_DIR / "mnist_confusion_matrix_percentage.png"
    plot_confusion_matrix_percentage(cm, test_acc_pct, cm_plot_path)


    # Print Confusion Matrix in percentages to console
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = (cm / row_sums) * 100.0
    print("\nConfusion Matrix (% per true class):")
    header = "     " + " ".join(f"{d:>6}" for d in range(10))
    print(header)
    print("-" * len(header))
    for i in range(10):
        row_str = f"D{i}: " + " ".join(f"{cm_pct[i, j]:6.1f}" for j in range(10))
        print(row_str)

    print("\nDone!")

if __name__ == "__main__":
    run_mnist_experiment()
