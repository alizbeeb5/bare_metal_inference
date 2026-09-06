import numpy as np
import pandas as pd
from pathlib import Path

FEATURE_COLUMNS = [
    "lead_time_days",
    "supplier_reliability",
    "distance_km",
    "season_q",
    "order_value",
    "item_category",
    "carrier",
    "weekend_flag",
]

def generate(n_rows=500, seed=42):
    rng = np.random.default_rng(seed)
    lead_time = rng.uniform(1.0, 30.0, n_rows)
    reliability = rng.uniform(0.5, 1.0, n_rows)
    distance = rng.uniform(50.0, 5000.0, n_rows)
    season = rng.integers(0, 4, n_rows)
    order_value = rng.uniform(10.0, 5000.0, n_rows)
    item_cat = rng.integers(0,5,n_rows)
    carrier = rng.integers(0,3,n_rows)
    weekend = rng.integers(0,2,n_rows)

    z = (
        -2.0
        +0.08*(lead_time - 15.0)
        -3.5 *(reliability - 0.85)
        +0.0004*(distance-1500.0)
        +0.15*season
        +0.0001*(order_value-1000.0)
        -0.1*item_cat
        -0.25*carrier
        +0.6*weekend
    )

    p = 1.0/(1.0+np.exp(-z)) #sigmoid
    delayed = (rng.uniform(0.0, 1.0, n_rows)<p).astype(np.int32)


    df = pd.DataFrame({
        "lead_time_days": lead_time.astype(np.float32),
        "supplier_reliability": reliability.astype(np.float32),
        "distance_km": distance.astype(np.float32),
        "season_q": season.astype(np.float32),
        "order_value": order_value.astype(np.float32),
        "item_category": item_cat.astype(np.float32),
        "carrier": carrier.astype(np.float32),
        "weekend_flag": weekend.astype(np.float32),
        "delayed": delayed,
    })


    return df


def main():
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate(5000,42)
    df.to_csv(out_dir/"shipments.csv", index = False)
    print(f"Generated {len(df)} rows-> {out_dir/'shipments.csv'}")
    print(f"Delay rate: {df['delayed'].mean():.3f}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()