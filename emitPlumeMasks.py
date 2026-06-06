import earthaccess
import os
import numpy as np 
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import json
import pprint
from shapely.geometry import shape #allows coordinate points to be converted into geometric shapes


with open("config.json") as f:
    config = json.load(f)

os.environ["EARTHDATA_USERNAME"] = config["username"]
os.environ["EARTHDATA_PASSWORD"] = config["password"]

earthaccess.login(strategy="environment")


bbox = (-105, 30, -101,33.5) #bounding box for Permian Basin
results = earthaccess.search_data( #collects data granules for all files that match the search criteria
    short_name="EMITL2BCH4PLM",
    bounding_box=bbox
    )



file = earthaccess.open([results[0]])
'''
with rasterio.open(file[0]) as src: #rasterio is library for geospatial raster data 
    print("Bands:", src.count)
    print("Shape:", src.shape)
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Transform:", src.transform)
'''

metadata = json.load(file[1])

feature = metadata["features"][0]
polygon = shape(feature["geometry"])

with rasterio.open(file[0]) as src:
    image = src.read(1)

    mask = rasterize(
        [(polygon, 1)],
        out_shape=src.shape,
        transform=src.transform,
        fill=0,
        dtype="uint8" #memory efficient data type for binary masks
    )
print(mask.shape)
print(mask.sum())

plt.figure(figsize=(8, 8))
plt.imshow(image, cmap="viridis")
plt.contour(mask, levels=[0.5], colors="red", linewidths=2) #overlays the mask on the image with red contours
plt.title("Mask Overlay")
plt.axis("off")

plt.show()






