
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import LineString
# =============================================================================
# Generate linestring segments from multilinestrings
# =============================================================================
file_path = "./data/output.gpkg"
gdf = gpd.read_file(file_path)
gdf.strandtype.hist()

gdf.geometry = gdf.geometry.explode().reset_index(level=1, drop=True)
gdf.geometry = gdf.geometry.apply(lambda x: loads(str(x)))


strandtype_data = []

for row in gdf[["geometry", "strandtype"]].itertuples(index=False):
    linestring = row.geometry
    strandtype = row.strandtype
    
    for i in range(len(linestring.coords) - 1):
        p1 = linestring.coords[i]
        p2 = linestring.coords[i + 1]
        strandtype_data.append((LineString([p1, p2]), strandtype))

data = {'Geometry': [item[0] for item in strandtype_data],
        'StrandType': [item[1] for item in strandtype_data]}

gdf = gpd.GeoDataFrame(data, geometry='Geometry')

gdf.crs = "EPSG:32633"
gdf.to_file("./data/strandlinje_segments.gpkg", driver="GPKG")




