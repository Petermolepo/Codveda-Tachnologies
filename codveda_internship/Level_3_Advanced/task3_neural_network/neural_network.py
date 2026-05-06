"""
Codveda Technology Internship
Level 3 - Task 3: Neural Networks with TensorFlow/Keras
Author: Peter Molepo
Description: Build, train and evaluate a feed-forward neural network for classification.
             Uses structured tabular data (customer churn) if MNIST is unavailable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
    "axes.titleweight": "bold",
})

# ─────────────────────────────────────────────
#  0. CHECK TENSORFLOW
# ─────────────────────────────────────────────

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
    print(f"\n  TensorFlow version : {tf.__version__}")
    tf.random.set_seed(42)
except ImportError:
    TF_AVAILABLE = False
    print("\n  TensorFlow not found – running in simulation mode.")
    print("  Install with: pip install tensorflow")


# ─────────────────────────────────────────────
#  1. LOAD / GENERATE DATASET
# ─────────────────────────────────────────────

def load_mnist():
    """Try to load MNIST from Keras datasets."""
    if not TF_AVAILABLE:
        return None, None, None, None, False

    try:
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
        X_test = X_test.reshape(-1, 784).astype("float32") / 255.0
        print(f"\n  MNIST loaded successfully")
        print(f"  Train : {X_train.shape}  |  Test : {X_test.shape}")
        print(f"  Classes : {np.unique(y_train)}")
        return X_train, X_test, y_train, y_test, True
    except Exception as e:
        print(f"\n  MNIST load failed ({e}) – using tabular fallback dataset.")
        return None, None, None, None, False


def create_tabular_dataset():
    """Structured dataset (Iris-style multi-class) as fallback."""
    np.random.seed(42)
    n_per_class = 400
    n_classes = 3

    data_list = []
    for cls in range(n_classes):
        means = np.random.uniform(cls * 2, cls * 2 + 2, 10)
        stds = np.random.uniform(0.5, 1.0, 10)
        X_cls = np.random.normal(means, stds, (n_per_class, 10))
        y_cls = np.full(n_per_class, cls)
        data_list.append((X_cls, y_cls))

    X = np.vstack([d[0] for d in data_list])
    y = np.concatenate([d[1] for d in data_list])

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Tabular dataset created")
    print(f"  Train : {X_train.shape}  |  Test : {X_test.shape}")
    print(f"  Classes : {np.unique(y_train)}")

    return X_train, X_test, y_train, y_test


def prepare_data():
    X_train, X_test, y_train, y_test, is_mnist = load_mnist()

    if X_train is None:
        X_train, X_test, y_train, y_test = create_tabular_dataset()
        dataset_name = "Tabular Multi-Class"
        input_dim = X_train.shape[1]
        n_classes = len(np.unique(y_train))
    else:
        dataset_name = "MNIST Handwritten Digits"
        input_dim = 784
        n_classes = 10

    return X_train, X_test, y_train, y_test, input_dim, n_classes, dataset_name


# ─────────────────────────────────────────────
#  2. BUILD NEURAL NETWORK
# ─────────────────────────────────────────────

def build_model(input_dim, n_classes, learning_rate=0.001):
    """Feed-forward neural network with dropout regularisation."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(n_classes, activation="softmax"),
    ], name="FeedForward_NN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def print_model_summary(model):
    print("\n" + "=" * 60)
    print("  STEP 2: NEURAL NETWORK ARCHITECTURE")
    print("=" * 60)
    model.summary()


# ─────────────────────────────────────────────
#  3. TRAIN
# ─────────────────────────────────────────────

def train_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=64):
    print("\n" + "=" * 60)
    print("  STEP 3: TRAINING")
    print("=" * 60)

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    return history


# ─────────────────────────────────────────────
#  4. EVALUATE
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, n_classes):
    print("\n" + "=" * 60)
    print("  STEP 4: EVALUATION")
    print("=" * 60)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=[f"Class {i}" for i in range(n_classes)]))

    return y_pred, acc


# ─────────────────────────────────────────────
#  5. HYPERPARAMETER EXPERIMENTS
# ─────────────────────────────────────────────

def hyperparameter_experiments(X_train, y_train, X_test, y_test, input_dim, n_classes):
    """Run quick experiments varying learning rate and batch size."""
    print("\n" + "=" * 60)
    print("  STEP 5: HYPERPARAMETER TUNING EXPERIMENTS")
    print("=" * 60)

    configs = [
        {"lr": 0.01,   "batch": 32,  "label": "LR=0.01, BS=32"},
        {"lr": 0.001,  "batch": 64,  "label": "LR=0.001, BS=64"},
        {"lr": 0.0005, "batch": 128, "label": "LR=0.0005, BS=128"},
    ]

    results = []
    for cfg in configs:
        m = build_model(input_dim, n_classes, cfg["lr"])
        h = m.fit(X_train, y_train,
                  epochs=15, batch_size=cfg["batch"],
                  validation_data=(X_test, y_test),
                  verbose=0,
                  callbacks=[
                      keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
                  ])
        _, acc = m.evaluate(X_test, y_test, verbose=0)
        val_acc = max(h.history["val_accuracy"])
        results.append({"Label": cfg["label"], "TestAcc": acc, "BestValAcc": val_acc})
        print(f"  {cfg['label']:<25}  Test Acc: {acc:.4f}  Best Val Acc: {val_acc:.4f}")

    return results


# ─────────────────────────────────────────────
#  6. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_results(history, y_test, y_pred, n_classes, hyp_results, dataset_name):
    # ── Plot 1: Training History ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History – {dataset_name}", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"], label="Train Accuracy", color="#3498DB", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", color="#E74C3C", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Curves")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss", color="#3498DB", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", color="#E74C3C", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Curves")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("plot_01_training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: plot_01_training_history.png")

    # ── Plot 2: Confusion Matrix ──────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes - 1)))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels([f"C{i}" for i in range(n_classes)])
    ax.set_yticklabels([f"C{i}" for i in range(n_classes)])
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=9, color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("plot_02_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_02_confusion_matrix.png")

    # ── Plot 3: Hyperparameter Results ────────────────────────
    if hyp_results:
        labels = [r["Label"] for r in hyp_results]
        test_accs = [r["TestAcc"] for r in hyp_results]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, test_accs, color=["#3498DB", "#E74C3C", "#27AE60"], alpha=0.85)
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Hyperparameter Tuning – Test Accuracy per Config")
        ax.set_ylim(min(test_accs) - 0.05, 1.0)
        for bar, v in zip(bars, test_accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.003, f"{v:.4f}", ha="center", fontsize=10)
        ax.tick_params(axis="x", rotation=10)
        plt.tight_layout()
        plt.savefig("plot_03_hyperparameter_tuning.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: plot_03_hyperparameter_tuning.png")


# ─────────────────────────────────────────────
#  7. SIMULATION MODE (no TF)
# ─────────────────────────────────────────────

def simulation_mode():
    """Show what the output would look like without TensorFlow installed."""
    print("\n" + "=" * 60)
    print("  SIMULATION MODE (TensorFlow not installed)")
    print("=" * 60)
    print("""
  To run this script fully, install TensorFlow:

      pip install tensorflow

  What this script does:
    1. Loads MNIST (or generates tabular data as fallback)
    2. Builds a 3-layer feed-forward neural network
    3. Trains with early stopping and LR scheduling
    4. Evaluates using accuracy, loss, confusion matrix
    5. Runs hyperparameter experiments (LR × batch size)
    6. Saves 3 plots: training curves, confusion matrix, HP tuning

  Expected output on MNIST:
    - Test accuracy : ~98%
    - Training time : ~2–3 minutes on CPU

  Architecture:
    Input (784) → Dense(256) → BN → Dropout(0.3)
               → Dense(128) → BN → Dropout(0.3)
               → Dense(64)  → Dropout(0.2)
               → Output(10) [softmax]
""")


# ─────────────────────────────────────────────
#  8. MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  CODVEDA INTERNSHIP | LEVEL 3 – TASK 3")
    print("  Neural Networks with TensorFlow / Keras")
    print("=" * 60)

    if not TF_AVAILABLE:
        simulation_mode()
        return

    X_train, X_test, y_train, y_test, input_dim, n_classes, dataset_name = prepare_data()

    model = build_model(input_dim, n_classes)
    print_model_summary(model)

    history = train_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=64)

    y_pred, final_acc = evaluate_model(model, X_test, y_test, n_classes)

    hyp_results = hyperparameter_experiments(
        X_train, y_train, X_test, y_test, input_dim, n_classes
    )

    print("\n  Generating visualisations...")
    plot_results(history, y_test, y_pred, n_classes, hyp_results, dataset_name)

    # Save model
    model.save("neural_network_model.keras")
    print("\n  Model saved: neural_network_model.keras")
    print(f"\n  Final Test Accuracy : {final_acc * 100:.2f}%")
    print("\n  Neural network pipeline complete.")


if __name__ == "__main__":
    main()
