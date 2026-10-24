import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.mean = np.mean(X, axis=0, keepdims=True)
        self.std = np.std(X, axis=0, keepdims=True)
        # Avoid division by zero
        self.std[self.std == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler must be fitted before inverse_transform.")
        X_scaled = np.asarray(X_scaled, dtype=float)
        return X_scaled * self.std + self.mean

def to_one_hot(y: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """
    Converts 1D label array to one-hot encoded matrix.
    Handles both 1-based (1, 2, 3) and 0-based (0, 1, 2) integer classes.
    """
    y = np.asarray(y, dtype=int).ravel()
    min_cls = np.min(y)
    if min_cls == 1:
        # Convert 1-based indexing to 0-based
        y = y - 1

    if num_classes is None:
        num_classes = int(np.max(y) + 1)

    one_hot = np.zeros((len(y), num_classes), dtype=float)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot

def print_class_distribution(y: np.ndarray, dataset_name: str = "Dataset") -> dict[int, tuple[int, float]]:
    """
    Calculates and prints the count and percentage of each class.
    Requirement: Make a count of each class in test set (x% of class1, y% of class2, etc.)
    """
    y_flat = np.asarray(y).ravel()
    unique_classes, counts = np.unique(y_flat, return_counts=True)
    total = len(y_flat)
    
    distribution_info = {}
    print(f"\n--- Class Distribution for [{dataset_name}] (Total: {total} samples) ---")
    for cls, count in zip(unique_classes, counts):
        pct = (count / total) * 100.0
        distribution_info[cls] = (count, pct)
        print(f"  Class {cls}: {count} samples ({pct:.2f}%)")
    print("----------------------------------------------------------------------\n")
    return distribution_info

def load_dataset(
    problem: str,
    dataset_name: str,
    sample_size: int,
    split: str = "train"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads dataset CSV from classification/ or regression/ folder.
    Returns:
        X: np.ndarray of features
        y: np.ndarray of targets / labels
    """
    file_path = PROJECT_ROOT / problem / f"data.{dataset_name}.{split}.{sample_size}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path)
    if problem == "classification":
        X = df[["x", "y"]].to_numpy(dtype=float)
        y = df["cls"].to_numpy(dtype=int)
        return X, y
    elif problem == "regression":
        X = df[["x"]].to_numpy(dtype=float)
        y = df[["y"]].to_numpy(dtype=float)
        return X, y
    else:
        raise ValueError(f"Unknown problem type: {problem}")
