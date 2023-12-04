
import os
import geopandas as gpd
from shapely.geometry import box
import rasterio

image_save_path = "./data/test"


geometries = []
filenames = []
for filename in os.listdir(image_save_path):
    if filename.endswith(".png"):
        image_path = os.path.join(image_save_path, filename)

        with rasterio.open(image_path) as src:
            bounds = src.bounds
            polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            geometries.append(polygon)
            filenames.append(filename)
gdf = gpd.GeoDataFrame({'geometry': geometries, "filename": filenames})


output_geopackage = "segmented_bboxes.gpkg"
gdf.crs = "EPSG:32633"
gdf.to_file("./data/" + output_geopackage, driver='GPKG')

print(f"Bounding boxes saved to {output_geopackage}")