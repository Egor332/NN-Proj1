import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_mnist, print_class_distribution

def main():
    """Loads the MNIST test set and prints the sample count and percentage of each digit class."""
    _, y_test = load_mnist(split="test")
    print_class_distribution(y_test, dataset_name="MNIST Test Set")

if __name__ == "__main__":
    main()
