"""Generate the synthetic sensor dataset used by ``stream_demo.ipynb``.

Each row is one reading from a machine: five numeric sensor values and a
``fault`` label (1 = fault). The label follows a non-linear rule of
temperature, vibration and pressure, with 5% label noise. About 3% of
sensor cells are missing (written as ``NA``) so the demo can show
streaming imputation. From row ``drift_at`` the fault rule changes
slightly (concept drift), as when a machine wears in.

Run from the repository root:

    python demo/generate_stream_data.py
"""

import argparse
from pathlib import Path

import numpy as np

COLUMNS = ["temperature", "vibration", "pressure", "humidity", "load", "fault"]


def generate(n_rows=6000, missing=0.03, noise=0.05, drift_at=4000, seed=42):
    """Return a (n_rows, 6) float array: five sensors and the fault label."""
    rng = np.random.default_rng(seed)
    temperature = rng.normal(70, 8, n_rows)
    vibration = rng.gamma(2.0, 1.5, n_rows)
    pressure = rng.normal(100, 15, n_rows)
    humidity = rng.uniform(20, 80, n_rows)       # unrelated to faults
    load = rng.normal(0.6, 0.15, n_rows)

    score = ((temperature - 70) / 8 + 0.8 * (vibration - 3) ** 2 / 4
             - 0.5 * (pressure - 100) / 15 + 1.5 * (load - 0.6) / 0.15)
    threshold = np.where(np.arange(n_rows) < drift_at, 1.0, 0.4)
    fault = score > threshold
    flip = rng.random(n_rows) < noise
    fault = np.where(flip, ~fault, fault).astype(float)

    data = np.column_stack([temperature, vibration, pressure, humidity, load,
                            fault])
    sensors = data[:, :5]
    sensors[rng.random(sensors.shape) < missing] = np.nan
    return data


def write_csv(path, data):
    """Write data with a header row; NaN becomes ``NA``."""
    with open(path, "w") as handle:
        handle.write(",".join(COLUMNS) + "\n")
        for row in data:
            cells = ["NA" if np.isnan(v) else f"{v:.4f}" for v in row[:5]]
            cells.append(str(int(row[5])))
            handle.write(",".join(cells) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "data"
                        / "sensor_stream.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = generate(n_rows=args.rows, seed=args.seed)
    write_csv(args.output, data)
    print(f"Wrote {len(data)} rows to {args.output} "
          f"({data[:, 5].mean():.1%} faults)")


if __name__ == "__main__":
    main()
