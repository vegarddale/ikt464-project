
import os
from osgeo import gdal
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS





image_dir = "./data/images/test_data/"
path_images = []
for file in os.listdir(image_dir):
    if file.endswith(".tif"):
        path_images.append(file)

for input_image in path_images:
    
    dataset = gdal.Open(os.path.join(image_dir, input_image))
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    
    segment_width = width // 2  
    segment_height = height // 4  
    
    original_geotransform = dataset.GetGeoTransform()
    original_crs = dataset.GetProjectionRef()
    
    original_crs = rasterio.crs.CRS.from_string(original_crs)
    
    image_save_path = "./data/test"
    os.makedirs(image_save_path, exist_ok=True)
    
    for x in range(0, width, segment_width):
        for y in range(0, height, segment_height):
            segment = dataset.ReadAsArray(x, y, segment_width, segment_height)
            segment = np.transpose(segment, (1, 2, 0))
            segment = segment.astype(np.uint8)
    
            plt.imshow(segment)
            plt.axis('off')
            plt.show()
    
            segment_image = Image.fromarray(segment)
            segment_image_path = os.path.join(image_save_path, f"{input_image}_{x}_{y}.png")
            segment_image.save(segment_image_path)
            
            adjusted_x = original_geotransform[0] + x * original_geotransform[1]
            adjusted_y = (original_geotransform[3] + (segment_height * original_geotransform[5])) + y * original_geotransform[5]
            
            segment_geotransform = from_origin(adjusted_x, adjusted_y, original_geotransform[1], original_geotransform[5])
    
            with rasterio.open(segment_image_path, 'w', driver='GTiff', width=segment_width, height=segment_height,
                               count=3, dtype='uint8', crs=original_crs, transform=segment_geotransform) as segment_ds:
                segment_ds.write(segment.transpose(2, 0, 1))
    
            # You can save the adjusted geotransform for this segment as needed
            with open(os.path.join(image_save_path, f"{input_image}_{x}_{y}_geotransform.txt"), "w") as f:
                f.write(str(segment_geotransform))
