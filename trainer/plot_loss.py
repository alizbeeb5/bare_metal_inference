import json
from pathlib import Path
import matplotlib.pyplot as plt

data_dir = Path(__file__).parent / "data"
with open(data_dir / "metrics.json") as f:
    metrics = json.load(f)

history = metrics["loss_history"]
epochs, losses = zip(*history)

plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, "b-", linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy loss")
plt.title(f"Training loss (final test accuracy: {metrics['test_accuracy']:.3f})")
plt.grid(True, alpha=0.3)
plt.tight_layout()

out = data_dir / "loss_curve.png"
plt.savefig(out, dpi=120)
print(f"Saved {out}")