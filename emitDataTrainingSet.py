import earthaccess 
import os 
import numpy as np 
import rasterio 
import matplotlib.pyplot as plt 
import csv 
import json 

earthaccess.login() 
bbox = (-105, 30, -101,33.5) 
#bounding box for Permian Basin 
results = earthaccess.search_data( #collects data granules for all PLM or positive methane samples according to NASA
    short_name="EMITL2BCH4PLM", bounding_box=bbox ) 

enh_results = earthaccess.search_data( #collects all methane enhancement samples (both positive and negative) 
    short_name="EMITL2BCH4ENH", bounding_box=bbox ) 

def bbox_intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 <= x1 or y2 <= y1:
        return 0

    return (x2 - x1) * (y2 - y1)


def bbox_area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def iou(a, b):
    inter = bbox_intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0

def polygon_to_bbox(points): #function to convert a polygon shape (what the methane enhancement samples are) to a bounding box - for simplicity sake to match the PLM samples
    lats = [p["Latitude"] for p in points] 
    lons = [p["Longitude"] for p in points] 
    return ( min(lons), # west 
             min(lats), # south 
             max(lons), # east 
             max(lats)  # north 
           ) 

def get_bbox(granule): #function to extract the bounding box from a granule's metadata, whether it's a rectangle or a polygon
    geom = granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"] 

    '''
    if "BoundingRectangles" in geom: 
        rect = geom["BoundingRectangles"][0] 
        return ( rect["WestBoundingCoordinate"], rect["SouthBoundingCoordinate"], rect["EastBoundingCoordinate"], rect["NorthBoundingCoordinate"] ) 
'''
    # CASE 2: polygon 
    if "GPolygons" in geom: 
        points = geom["GPolygons"][0]["Boundary"]["Points"] 
        return polygon_to_bbox(points) 

    return None 

plm_boxes = [] #list of all known plumes
for g in results: 
    bbox = get_bbox(g) 
    if bbox: 
        plm_boxes.append(bbox) 

#print(results[0].data_links()) 
#print(results[0]["umm"]["DataGranule"]) 

with open("training.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_url", "label"]) #creating csv file

    for granule in enh_results:
        enh_bbox = get_bbox(granule)
        if not enh_bbox:
            continue

        is_positive = any(
            iou(enh_bbox, p) > 0.02   # try 0.01–0.05 range
            for p in plm_boxes
        )

        label = 1 if is_positive else 0

        tif_link = next( #selecting the first .tif link from the granule's data links, which is the image we want to use for training
            (l for l in granule.data_links() if l.endswith(".tif")),
            None
        )

        if tif_link:
            writer.writerow([tif_link, label])

from collections import Counter

with open("training.csv") as f:
    next(f)
    print(Counter(int(line.split(",")[1]) for line in f))

''' 
with open("positives.csv", mode='w', newline='') as file: 
    writer = csv.writer(file) 
    writer.writerow(["image_url", "label"]) 

for granule in results: 
    data_links = granule.data_links() 
    for link in data_links: 
        if link.endswith('.tif'): 
            writer.writerow([link, 1]) 

print("Done!") ''' '''' files = earthaccess.open([results[0]]) with rasterio.open(files[0]) as src: #rasterio is library for geospatial raster data data = src.read(1) # Read the first band plt.imshow(data, cmap='viridis') #viridis is a color map that is perceptually uniform and is often used for scientific data visualization plt.colorbar(label='CH4 Concentration') plt.title('EMIT L2B CH4 Enhancement') plt.axis("off") plt.show() print("Min:", np.min(data)) print("Max:", np.max(data)) print("Mean:", np.mean(data)) '''