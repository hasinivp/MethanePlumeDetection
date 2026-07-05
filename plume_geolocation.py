import rasterio

def pixel_to_latlon(row_256, col_256, orig_height, orig_width, transform):
    # Calculate the pixel coordinates in the original image
    scale = min(256 / orig_height, 256 / orig_width)
    new_height = int(orig_height * scale)
    new_width = int(orig_width * scale)
    pad_top = (256 - new_height) // 2
    pad_left = (256 - new_width) // 2
    row_unpadded = row_256 - pad_top
    col_unpadded = col_256 - pad_left
    orig_row = row_unpadded / scale
    orig_col = col_unpadded / scale
    lon, lat = rasterio.transform.xy(transform, orig_row, orig_col) #convert pixel coordinates to geographic coordinates using the affine transform
    return lat, lon