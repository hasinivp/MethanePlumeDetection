#data info resource: https://catalog.data.gov/dataset/emit-l2b-estimated-methane-plume-complexes-60-m-v001

import earthaccess
import os
import numpy as np 
import rasterio
import matplotlib.pyplot as plt
import csv


earthaccess.login()

bbox = (-105, 30, -101,33.5) #bounding box for Permian Basin
results = earthaccess.search_data( #collects data granules for all files that match the search criteria
    short_name="EMITL2BCH4PLM",
    bounding_box=bbox
    )

print(len(results))
#print (results[0])
print(results[0].data_links())

with open("positives.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_url", "label"])

    for granule in results:
        data_links = granule.data_links()
        for link in data_links:
            if link.endswith('.tif'):
                writer.writerow([link, 1])

print("Done!")

''''
files = earthaccess.open([results[0]])
with rasterio.open(files[0]) as src: #rasterio is library for geospatial raster data 
    data = src.read(1)  # Read the first band



plt.imshow(data, cmap='viridis') #viridis is a color map that is perceptually uniform and is often used for scientific data visualization
plt.colorbar(label='CH4 Concentration')
plt.title('EMIT L2B CH4 Enhancement')
plt.axis("off")
plt.show()

print("Min:", np.min(data))
print("Max:", np.max(data))
print("Mean:", np.mean(data))
'''