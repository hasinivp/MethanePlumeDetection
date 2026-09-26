"""
Stage 1 - File 5: Run inference and produce bounding boxes for Stage 2.

Pipeline:
    stage1_stack.tif + stage1_model.h5
        -> per-pixel anomaly heatmaps
        -> threshold to candidate plume pixels
        -> cluster nearby pixels into distinct plumes
        -> one padded bounding box per cluster
        -> bounding_boxes.json   (this is what Stage 2 reads)

Usage:
    python 5_infer.py                      # defaults
    python 5_infer.py stage1_stack.tif 0.5 # <tif> <threshold>

Tuning (top of file):
    THRESHOLD          higher -> fewer, stronger detections
    CLUSTER_RADIUS_DEG larger -> more pixels merge -> fewer boxes
    PADDING_DEG        margin added around each box
    MIN_PIXELS         drop clusters smaller than this (noise filter)
"""

import sys
import json
import numpy as np
import rasterio
import tensorflow as tf
from scipy import ndimage

TIF_FILE = "stage1_stack.tif"
MODEL_FILE = "stage1_model.h5"
OUTPUT_FILE = "bounding_boxes.json"

THRESHOLD = 0.5
CLUSTER_RADIUS_DEG = 0.08
PADDING_DEG = 0.03
MIN_PIXELS = 2

PATCH = 11
HALF = PATCH // 2


def load_timesteps(tif_file):
    with rasterio.open(tif_file) as src:
        data = src.read()
        n_bands = src.count
        H, W = src.height, src.width
        transform = src.transform

    if n_bands % 4 == 0:
        vars_per_day, var_list = 4, ["ch4", "u", "v", "ogim"]
    elif n_bands % 3 == 0:
        vars_per_day, var_list = 3, ["ch4", "u", "v"]
    else:
        raise SystemExit(f"ERROR: {n_bands} bands not divisible by 3 or 4")

    timesteps = {}
    for band_idx in range(n_bands):
        day = band_idx // vars_per_day
        var = var_list[band_idx % vars_per_day]
        if var == "ogim":
            continue
        timesteps.setdefault(day, {})[var] = data[band_idx]

    return timesteps, H, W, transform


def cluster_points(points, radius_deg):
    """Greedy spatial clustering by running-centroid distance."""
    clusters, centroids = [], []
    for p in points:
        placed = False
        for i, (clat, clon) in enumerate(centroids):
            if np.hypot(p["lat"] - clat, p["lon"] - clon) < radius_deg:
                clusters[i].append(p)
                lats = [q["lat"] for q in clusters[i]]
                lons = [q["lon"] for q in clusters[i]]
                centroids[i] = (np.mean(lats), np.mean(lons))
                placed = True
                break
        if not placed:
            clusters.append([p])
            centroids.append((p["lat"], p["lon"]))
    return clusters


def main():
    tif_file = sys.argv[1] if len(sys.argv) > 1 else TIF_FILE
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else THRESHOLD

    print(f"Inference on {tif_file}  (threshold={threshold})")

    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    timesteps, H, W, transform = load_timesteps(tif_file)
    time_keys = sorted(timesteps.keys())
    print(f"Timesteps: {len(time_keys)}")

    # Build sequences for every valid patch center
    X_all, coords = [], []
    for r in range(HALF, H - HALF):
        for c in range(HALF, W - HALF):
            seq = []
            for t in time_keys:
                f = timesteps[t]
                if not all(k in f for k in ("ch4", "u", "v")):
                    continue
                seq.append(np.stack([
                    f["ch4"][r-HALF:r+HALF+1, c-HALF:c+HALF+1],
                    f["u"][r-HALF:r+HALF+1, c-HALF:c+HALF+1],
                    f["v"][r-HALF:r+HALF+1, c-HALF:c+HALF+1],
                ], axis=-1))
            if len(seq) == len(time_keys):
                X_all.append(np.stack(seq))
                coords.append((r, c))

    X_all = np.array(X_all, dtype=np.float32)
    print(f"Patches: {len(X_all)}")

    for i in range(3):
        m, s = X_all[..., i].mean(), X_all[..., i].std()
        X_all[..., i] = (X_all[..., i] - m) / (s + 1e-8)

    heatmaps = np.squeeze(model.predict(X_all, verbose=0), axis=-1)
    print(f"Heatmap range: [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")

    # Threshold -> candidate pixels (in lat/lon)
    points = []
    for heatmap, (r_patch, c_patch) in zip(heatmaps, coords):
        labeled, n = ndimage.label(heatmap > threshold)
        for comp_id in range(1, n + 1):
            mask = labeled == comp_id
            intensity = float(heatmap[mask].mean())
            yy, xx = ndimage.center_of_mass(mask)
            row = int(np.clip(r_patch - HALF + yy, 0, H - 1))
            col = int(np.clip(c_patch - HALF + xx, 0, W - 1))
            lon = transform.c + col * transform.a
            lat = transform.f + row * transform.e
            points.append({"lat": float(lat), "lon": float(lon),
                           "probability": intensity})

    print(f"Candidate pixels: {len(points)}")

    clusters = cluster_points(points, CLUSTER_RADIUS_DEG)

    boxes = []
    for cluster in clusters:
        if len(cluster) < MIN_PIXELS:
            continue
        lats = [p["lat"] for p in cluster]
        lons = [p["lon"] for p in cluster]
        probs = [p["probability"] for p in cluster]
        boxes.append({
            "bbox": {
                "lat_min": float(min(lats) - PADDING_DEG),
                "lat_max": float(max(lats) + PADDING_DEG),
                "lon_min": float(min(lons) - PADDING_DEG),
                "lon_max": float(max(lons) + PADDING_DEG),
            },
            "center": {"lat": float(np.mean(lats)),
                       "lon": float(np.mean(lons))},
            "probability": float(max(probs)),
            "n_pixels": len(cluster),
        })

    boxes.sort(key=lambda b: b["probability"], reverse=True)
    for i, b in enumerate(boxes):
        b["cluster_id"] = i

    print(f"\nBounding boxes: {len(boxes)}")
    for b in boxes:
        bb = b["bbox"]
        print(f"  Box {b['cluster_id']}: "
              f"lat[{bb['lat_min']:.3f},{bb['lat_max']:.3f}] "
              f"lon[{bb['lon_min']:.3f},{bb['lon_max']:.3f}] "
              f"prob={b['probability']:.3f} px={b['n_pixels']}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(boxes, f, indent=2)
    print(f"\nSaved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
