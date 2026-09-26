"""
Stage 1 - File 3: Turn stage1_stack.tif into training tensors.

Reads the multi-band GeoTIFF by BAND ORDER (not band names, which GEE exports
don't preserve reliably). Auto-detects whether the file has 3 vars/day
(ch4,u,v) or 4 vars/day (ch4,u,v,ogim) and ignores OGIM.

Produces sliding 11x11 patches through time:
    X : (N, T, 11, 11, 3)   sequence of [ch4, u, v] patches
    y : (N, 11, 11)         CH4 anomaly heatmap at the final timestep
                            (anomaly = ch4 - time-mean baseline, clamped >= 0)

Saves X_train / X_val / y_train / y_val as .npy for file 4.
"""

import numpy as np
import rasterio
from sklearn.model_selection import train_test_split

TIF_FILE = "stage1_stack.tif"
PATCH = 11
HALF = PATCH // 2


def load_timesteps(tif_file):
    with rasterio.open(tif_file) as src:
        data = src.read()
        n_bands = src.count
        H, W = src.height, src.width

    print(f"Loaded {tif_file}: {H} x {W}, {n_bands} bands")

    if n_bands % 4 == 0:
        vars_per_day, var_list = 4, ["ch4", "u", "v", "ogim"]
        print("Format: 4 vars/day (ch4,u,v,ogim) - OGIM will be ignored")
    elif n_bands % 3 == 0:
        vars_per_day, var_list = 3, ["ch4", "u", "v"]
        print("Format: 3 vars/day (ch4,u,v)")
    else:
        raise SystemExit(f"ERROR: {n_bands} bands not divisible by 3 or 4")

    timesteps = {}
    for band_idx in range(n_bands):
        day = band_idx // vars_per_day
        var = var_list[band_idx % vars_per_day]
        if var == "ogim":
            continue
        timesteps.setdefault(day, {})[var] = data[band_idx]

    return timesteps, H, W


def main():
    timesteps, H, W = load_timesteps(TIF_FILE)
    time_keys = sorted(timesteps.keys())
    print(f"Timesteps: {len(time_keys)}  ->  {time_keys}")

    # Baseline = mean CH4 across time (the "background")
    ch4_baseline = np.zeros((H, W))
    count = 0
    for t in time_keys:
        if "ch4" in timesteps[t]:
            ch4_baseline += timesteps[t]["ch4"]
            count += 1
    ch4_baseline /= max(count, 1)

    print("CH4 baseline: "
          f"min={ch4_baseline.min():.4f} "
          f"mean={ch4_baseline.mean():.4f} "
          f"max={ch4_baseline.max():.4f}")

    X, y = [], []

    for r in range(HALF, H - HALF):
        for c in range(HALF, W - HALF):
            seq = []
            final_anom = None

            for t_idx, t in enumerate(time_keys):
                frame = timesteps[t]
                if not all(k in frame for k in ("ch4", "u", "v")):
                    continue

                patch = np.stack([
                    frame["ch4"][r-HALF:r+HALF+1, c-HALF:c+HALF+1],
                    frame["u"][r-HALF:r+HALF+1, c-HALF:c+HALF+1],
                    frame["v"][r-HALF:r+HALF+1, c-HALF:c+HALF+1],
                ], axis=-1)
                seq.append(patch)

                if t_idx == len(time_keys) - 1:
                    ch4_patch = frame["ch4"][r-HALF:r+HALF+1, c-HALF:c+HALF+1]
                    base_patch = ch4_baseline[r-HALF:r+HALF+1, c-HALF:c+HALF+1]
                    final_anom = np.maximum(ch4_patch - base_patch, 0)

            if len(seq) != len(time_keys) or final_anom is None:
                continue

            X.append(np.stack(seq))
            y.append(final_anom)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")

    if len(y) == 0:
        raise SystemExit("ERROR: no samples extracted (image too small?)")

    print("y stats: "
          f"min={y.min():.4f} mean={y.mean():.4f} max={y.max():.4f} "
          f"(>0: {(y > 0).mean() * 100:.1f}% of pixels)")

    # Normalize each input feature to ~zero mean, unit std
    for i in range(3):
        m, s = X[..., i].mean(), X[..., i].std()
        X[..., i] = (X[..., i] - m) / (s + 1e-8)

    # Scale labels to [0,1]
    y_max = y.max()
    if y_max > 0:
        y = y / y_max

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"\nTrain: X{X_train.shape} y{y_train.shape}")
    print(f"Val:   X{X_val.shape} y{y_val.shape}")

    np.save("X_train.npy", X_train)
    np.save("X_val.npy", X_val)
    np.save("y_train.npy", y_train)
    np.save("y_val.npy", y_val)

    print("\nSaved X_train.npy, X_val.npy, y_train.npy, y_val.npy")


if __name__ == "__main__":
    main()
