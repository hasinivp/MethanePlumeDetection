import earthaccess
import os
import numpy as np 
import rasterio
import matplotlib.pyplot as plt


earthaccess.login()

bbox = (-105, 30, -101,33.5) #bounding box for Permian Basin
results = earthaccess.search_data( #collects data granules for all files that match the search criteria
    short_name="EMITL2BCH4ENH",
    bounding_box=bbox
    )

print(len(results))



files = earthaccess.open([results[0]])
print(files)
print(type(files[0]))


with rasterio.open(files[0]) as src:
    data = src.read(1)  # Read the first band



plt.imshow(data, cmap='viridis') #viridis is a color map that is perceptually uniform and is often used for scientific data visualization
plt.colorbar(label='CH4 Concentration')
plt.title('EMIT L2B CH4 Enhancement')
plt.axis("off")
plt.show()