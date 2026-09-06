import json
import struct
from pathlib import Path

MAGIC = b"BMIE"
VERSION = 1


def main() -> None:
    here = Path(__file__).parent
    data_dir = here / "data"
    with open(data_dir / "scaler.json", "rb") as f:
        scalar = json.load(f)
    with open(data_dir / "weights.json", "rb") as f:
        weights_obj = json.load(f)
    mean = scalar["mean"]
    std = scalar["std"]
    weights = weights_obj["weights"]
    bias = weights_obj["bias"]
    assert len(mean) == len(std) == len(weights)
    n = len(mean)

    out_path = here.parent / ("model.bin")
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III",VERSION, n, 0))
        f.write(struct.pack(f"<{n}f", *mean))
        f.write(struct.pack(f"<{n}f", *std))
        f.write(struct.pack(f"<{n}f", *weights))
        f.write(struct.pack("<f", bias))


    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"   n_features: {n}")
    print(f"   bias: {bias:.4f}")
    print(f"   first 3 weights: {weights[:3]}")


if __name__ == "__main__":
    main()