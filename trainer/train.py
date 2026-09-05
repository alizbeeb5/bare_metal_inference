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

def sigmoid(z):
    return np.where(z>=0, 1.0/(1.0+np.exp(-z)),np.exp(-z)/(1.0+np.exp(z)))

def fit(X, y, lr, epochs, l2):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for epoch in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        error = y - p
        grad_w = (X.T @ error) / n+l2*w
        grad_b = error.mean()
        w -= lr*grad_w
        b -= lr*grad_b
        if epoch % 50 == 0:
            loss = -np.mean(y*np.log(p+1e-12)+(1-y)*np.log( 1 - p + 1e-12))
            acc  = ((p>=0.5).astype(int)==y).mean()
            print(f"epoch: {epoch:4d}, loss: {loss:.4f}, acc: {acc:.4f}")