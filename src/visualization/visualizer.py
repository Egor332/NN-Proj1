import numpy as np
import matplotlib.pyplot as plt

class NetworkVisualizer:

    def __init__(self, style: str = "default"):
        try:
            plt.style.use(style)
        except Exception:
            pass

    def visualise_loss(
        self,
        loss_history: list[float],
        val_loss_history: list[float] | None = None,
        title: str = "Loss x Epoch",
        save_path: str | None = None,
        show: bool = True
    ):
        plt.figure(figsize=(8, 5))
        epochs = range(1, len(loss_history) + 1)
        plt.plot(epochs, loss_history, label="Train Loss", color="royalblue", linewidth=2)

        if val_loss_history is not None and len(val_loss_history) > 0:
            val_epochs = range(1, len(val_loss_history) + 1)
            plt.plot(val_epochs, val_loss_history, label="Validation Loss", color="crimson", linestyle="--", linewidth=2)

        plt.title(title, fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Loss function value", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def visualise_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model=None,
        title: str = "Results of regression model training (R -> R)",
        save_path: str | None = None,
        show: bool = True
    ):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        x_plot = X_arr.ravel()
        y_plot = y_arr.ravel()

        plt.figure(figsize=(9, 6))
        plt.scatter(x_plot, y_plot, color="dodgerblue", alpha=0.6, edgecolors="k", s=35, label="Data points")

        if model is not None:
            x_min, x_max = x_plot.min(), x_plot.max()
            margin = (x_max - x_min) * 0.05
            x_dense = np.linspace(x_min - margin, x_max + margin, 500).reshape(-1, 1)

            try:
                y_pred_dense = model.predict(x_dense)
                plt.plot(x_dense.ravel(), y_pred_dense.ravel(), color="crimson", linewidth=2.5, label="MLP aproximation")
            except Exception as e:
                print(f"Error: {e}")

        plt.title(title, fontsize=13, fontweight="bold")
        plt.xlabel("X", fontsize=11)
        plt.ylabel("Y", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def visualise_classification(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model=None,
        title: str = "Classification training results",
        resolution: int = 200,
        save_path: str | None = None,
        show: bool = True
    ):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if X_arr.shape[1] != 2:
            raise ValueError()

        if y_arr.ndim > 1 and y_arr.shape[1] > 1:
            y_labels = np.argmax(y_arr, axis=1)
        else:
            y_labels = y_arr.ravel().astype(int)

        plt.figure(figsize=(9, 7))

        if model is not None:
            x_min, x_max = X_arr[:, 0].min(), X_arr[:, 0].max()
            y_min, y_max = X_arr[:, 1].min(), X_arr[:, 1].max()
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1

            xx, yy = np.meshgrid(
                np.linspace(x_min - x_margin, x_max + x_margin, resolution),
                np.linspace(y_min - y_margin, y_max + y_margin, resolution)
            )
            grid = np.c_[xx.ravel(), yy.ravel()]

            try:
                preds = model.predict(grid)
                if preds.ndim > 1 and preds.shape[1] > 1:
                    zz = np.argmax(preds, axis=1)
                else:
                    p = preds.ravel()
                    zz = (p >= 0.5 if (p.min() >= 0.0 and p.max() <= 1.0) else (p >= 0.0)).astype(int)

                zz = zz.reshape(xx.shape)
                plt.contourf(xx, yy, zz, alpha=0.25, cmap="tab10")
                plt.contour(xx, yy, zz, colors="black", linewidths=0.8, alpha=0.7)
            except Exception as e:
                print(f"Can not find decision borders: {e}")

        unique_classes = np.unique(y_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_classes))))

        for idx, cls in enumerate(unique_classes):
            mask = (y_labels == cls)
            plt.scatter(
                X_arr[mask, 0],
                X_arr[mask, 1],
                color=colors[idx % len(colors)],
                edgecolors="black",
                s=45,
                label=f"Class {cls}",
                alpha=0.85
            )

        plt.title(title, fontsize=13, fontweight="bold")
        plt.xlabel("Feature 1 (x)", fontsize=11)
        plt.ylabel("Feature 2 (y)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def visualise_weights(
        self,
        weight_snapshots: list[tuple[int, list[np.ndarray]]],
        layer_idx: int | None = None,
        max_trajectories: int = 15,
        title: str = "Weights modification during training",
        save_path: str | None = None,
        show: bool = True
    ):
        if not weight_snapshots:
            print("No weights recived")
            return

        epochs = [snap[0] for snap in weight_snapshots]
        num_params = len(weight_snapshots[0][1])

        if layer_idx is not None:
            if layer_idx < 0 or layer_idx >= num_params:
                raise ValueError(f"Wrong parameter index: {layer_idx}. Available: 0..{num_params - 1}")

            param_data = np.array([snap[1][layer_idx].ravel() for snap in weight_snapshots])
            plt.figure(figsize=(9, 6))

            num_weights = param_data.shape[1]
            weights_to_plot = min(num_weights, max_trajectories)
            for w in range(weights_to_plot):
                plt.plot(epochs, param_data[:, w], label=f"Waga {w+1}", linewidth=1.5)

            plt.title(f"{title} (Layer {layer_idx})", fontsize=13, fontweight="bold")
            plt.xlabel("Epoch", fontsize=11)
            plt.ylabel("Weights value", fontsize=11)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(fontsize=9, loc="best")
            plt.tight_layout()

        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            for p_idx in range(num_params):
                norms = [np.linalg.norm(snap[1][p_idx]) for snap in weight_snapshots]
                ax1.plot(epochs, norms, label=f"Parametr {p_idx+1}", linewidth=2)

            ax1.set_title("L2 norm", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Epoch", fontsize=11)
            ax1.set_ylabel("L2", fontsize=11)
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend(fontsize=9)

            first_param = np.array([snap[1][0].ravel() for snap in weight_snapshots])
            num_weights = first_param.shape[1]
            weights_to_plot = min(num_weights, max_trajectories)

            for w in range(weights_to_plot):
                ax2.plot(epochs, first_param[:, w], label=f"Weight {w+1}", linewidth=1.4)

            ax2.set_title(f"Single weights trajectory (Parametr 1, max {weights_to_plot})", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Epoch", fontsize=11)
            ax2.set_ylabel("Weight value", fontsize=11)
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.legend(fontsize=8, loc="best")

            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
            plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()
