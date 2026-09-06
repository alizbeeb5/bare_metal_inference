import json
from pathlib import Path
import numpy as np
import pandas as pd
from generate_data import FEATURE_COLUMNS

LEARNING_RATE = 0.1
EPOCHS = 500
L2 = 1e-4
TEST_FRACTION = 0.2
SEED = 42

def sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid: compute elementwise to avoid overflow
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)
    mask = z >= 0
    # for z >= 0: 1 / (1 + exp(-z))
    out[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
    # for z < 0: exp(z) / (1 + exp(z))
    neg = ~mask
    out[neg] = np.exp(z[neg]) / (1.0 + np.exp(z[neg]))
    return out


def fit(X, y, lr, epochs, l2):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    history = []

    for epoch in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        error = p - y
        grad_w = (X.T @ error) / n + l2 * w
        grad_b = error.mean()
        w -= lr * grad_w
        b -= lr * grad_b

        if epoch % 5 == 0:
            loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            history.append((epoch, float(loss)))

    return w, b, history


def main() -> None:

    #read the data
    data_dir = Path(__file__).parent / "data"
    df = pd.read_csv(data_dir / "shipments.csv")
    print(f"Loaded {len(df)} rows")
    X_raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["delayed"].to_numpy(dtype=np.float64)

    #split
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * TEST_FRACTION)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train_raw, y_train = X_raw[train_idx], y[train_idx]
    X_test_raw,  y_test  = X_raw[test_idx],  y[test_idx]

    #compute mean for training data only
    mean = X_train_raw.mean(axis=0)
    std  = X_train_raw.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)

    #standardize both
    X_train = (X_train_raw - mean) / std
    X_test  = (X_test_raw  - mean) / std


    #training
    print("Fitting logistic regression (gradient descent)...")
    w, b, history = fit(X_train, y_train, LEARNING_RATE, EPOCHS, L2)

    #eval
    p_test = sigmoid(X_test @ w+b)
    test_acc = ((p_test>=0.5).astype(int)==y_test).mean()
    print(f"Test accuracy: {test_acc:.4f}")

    #save as json
    with open(data_dir / "scaler.json", "w") as f:
        json.dump({
            "mean":         mean.tolist(),
            "std":          std.tolist(),
            "feature_order": FEATURE_COLUMNS,
        }, f, indent=2)

    with open(data_dir / "weights.json", "w") as f:
        json.dump({
            "weights": w.tolist(),
            "bias":    float(b),
        }, f, indent=2)

    with open(data_dir / "metrics.json", "w") as f:
        json.dump({
            "test_accuracy": float(test_acc),
            "n_train":       int(len(X_train)),
            "n_test":        int(len(X_test)),
            "epochs":        EPOCHS,
            "lr":            LEARNING_RATE,
            "l2":            L2,
            "loss_history":  history,  # NEW
        }, f, indent=2)

    print(f"Wrote scaler.json, weights.json, metrics.json to {data_dir}")

if __name__ == "__main__":
    main()