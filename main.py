import earthaccess
import os
import json
import numpy as np
import rasterio
import torch
import cv2
from rasterio.features import rasterize, shapes
from unet import UNet
from sklearn.decomposition import PCA



def plume_polygon(mask, transform):
    
    mask = mask.astype(np.uint8)
    polygons = []
    for geom, value in shapes(mask, mask > 0, transform=transform):
        if value == 1:
            polygons.append(geom)

    return polygons

def get_centroid(mask):
    rows, cols = np.where(mask > 0)
    if len(rows) == 0 or len(cols) == 0:
        return None, None
    center_row = rows.mean()
    center_col = cols.mean()
    return center_row, center_col

def orientation_difference(plume_angle, wind_angle):
    if(plume_angle is None):
        print("Skip")
        return None
    else:
        diff1 = abs(plume_angle - wind_angle)
        diff1 = min(diff1, 360 - diff1)

        opposite = (plume_angle + 180) % 360

        diff2 = abs(opposite - wind_angle)
        diff2 = min(diff2, 360 - diff2)

        return min(diff1, diff2)

def plume_pca(mask):
    rows, cols = np.where(mask > 0)
    points = np.column_stack((cols, rows)) #Creates xy coordinates
    #PCA essentially finds line of best fit through the plume
    pca = PCA(2) #find the two main directions of the 2D point cloud
    pca.fit(points)
    ratio = pca.explained_variance_ratio_[0]
    if ratio<.6:
        print("orientation unreliable")
        return None, ratio
    else:
        direction_vector = pca.components_[0]
        dx = direction_vector[0]
        dy = direction_vector[1]
        plume_angle = np.degrees(np.arctan2(dy, dx)) % 360
        print(plume_angle)
        return plume_angle, ratio




def resize_image(tif_path):
    with rasterio.open(tif_path) as src:
        img = src.read(1)
        transform = src.transform #allows pixels to be matched to actual locations
        crs = src.crs
        orig_height = src.height 
        orig_width = src.width
    img[np.isclose(img, -9999, atol=5)] = 0
    img_height = img.shape[0]
    img_width = img.shape[1]
    scale = min(256 / img_height, 256 / img_width)
    new_height = int(img_height * scale)
    new_width = int(img_width * scale)
    image_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    pad_top = (256 - new_height) // 2
    pad_bottom = 256 - new_height - pad_top
    pad_left = (256 - new_width) // 2
    pad_right = 256 - new_width - pad_left
    image_padded = cv2.copyMakeBorder(image_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    assert image_padded.shape == (256, 256)
    image_tensor = torch.tensor(image_padded, dtype = torch.float32).unsqueeze(0).unsqueeze(0)
    return image_tensor, scale, pad_top, pad_left, transform, crs, orig_height, orig_width

def restore_original_mask(pred_binary, scale, pad_top, pad_left, #restores the mask to the original dimensions of the EMIT image
                          orig_height, orig_width):

    mask = pred_binary.squeeze().numpy().astype(np.uint8)

    new_height = int(orig_height * scale)
    new_width = int(orig_width * scale)

    # Remove the padding added before inference
    mask = mask[
        pad_top:pad_top + new_height,
        pad_left:pad_left + new_width
    ]

    # Resize back to the original EMIT dimensions
    mask = cv2.resize(
        mask,
        (orig_width, orig_height),
        interpolation=cv2.INTER_NEAREST
    )

    return mask

def save_geotiff(mask, transform, crs, output_file): #save mask as geotiff

    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
    ) as dst:

        dst.write(mask, 1)

print("start")
with open("config.json") as f:
    config = json.load(f)

os.environ["EARTHDATA_USERNAME"] = config["username"]
os.environ["EARTHDATA_PASSWORD"] = config["password"]


earthaccess.login(strategy="environment")
print("logged in")

model = UNet()
model.load_state_dict(torch.load("unet_epoch_1.pth"))
model.eval()
with open("detections.json") as f:
    detections = json.load(f)   
for box_idx, detection in enumerate(detections):
    lats = detection["lat"]
    lons = detection["lon"]
    lat1 = lats[0]
    lon1 = lons[0]
    lat2 = lats[1]
    lon2 = lons[1]
    buffer = .02
    bbox = (lon1 - buffer, lat1 - buffer, lon2 + buffer, lat2 + buffer) #bounding box for Permian Basin 
    results = earthaccess.search_data( #collects data granules for all files that match the search criteria
        short_name="EMITL2BCH4ENH",
        bounding_box=bbox,
        count = 10
        )


    print("Results obtained")
    
    all_plumes = []
    from ERA5_data_test import retrive_era5
    for i, result in enumerate(results):
        files = earthaccess.open([result])
        print(f"\n===== ENH Scene {i} =====")

        timestamp = (results[i]["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
        from datetime import datetime

        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        year = str(dt.year)
        month = f"{dt.month:02d}"
        day = f"{dt.day:02d}"
        hour = f"{dt.hour:02d}:00"

        files = earthaccess.open([results[i]])
        image_tensor, scale, pad_top, pad_left, transform, crs, orig_height, orig_width = resize_image(files[0])

        with torch.no_grad():
            output = model(image_tensor)

        predicted_mask = torch.sigmoid(output)
        pred_binary = (predicted_mask>.5).float()
        pred_original = restore_original_mask(pred_binary, scale, pad_top, pad_left, orig_height, orig_width)
        pixel_area = int(pred_original.sum())
        polygons = plume_polygon(pred_original, transform)
        center_row, center_col = get_centroid(pred_original)
        print(pred_original.sum())
        from plume_geolocation import pixel_to_latlon
        lat, lon = pixel_to_latlon(center_row, center_col, orig_height, orig_width, transform)

    
        u, v = retrive_era5(year, month, day, hour, lat, lon)
        print("U, V:", u, v)
        print("lat/lon:", lat, lon)
        print("ERA5 time:", year, month, day, hour)

        wind_angle = np.degrees(np.arctan2(v, u)) % 360
        print("Wind angle: ")
        print(wind_angle)
        print("Plume angle:")
        plume_angle, shape_score = plume_pca(pred_original)
        
        difference = orientation_difference(plume_angle, wind_angle)
        mask_filename = f"box_{box_idx}_plume_{i}.tif"
        save_geotiff(pred_original, transform, crs, mask_filename)
        plume_record = {
        "timestamp": timestamp,

        "centroid_lat": float(lat),
        "centroid_lon": float(lon),

        "wind_u": float(u),
        "wind_v": float(v),

        "wind_angle": float(wind_angle),

        "plume_angle":
            None if plume_angle is None
            else float(plume_angle),

        "orientation_difference":
            None if difference is None
            else float(difference),

        "shape_score": float(shape_score),

        "pixel_area": pixel_area,

        "polygon": polygons,
        
        "mask_filename": mask_filename,
        }

        all_plumes.append(plume_record)

with open("plume_results.json", "w") as f:
    json.dump(all_plumes, f, indent=4)

print("Saved plume_results.json")