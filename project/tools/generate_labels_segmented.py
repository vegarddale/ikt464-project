
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as ET
import os
import numpy as np
import rasterio
from shapely.geometry import Polygon, box, LineString
import rasterio
import cv2
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
from pyproj import Proj, transform

# =============================================================================
# Get bounding box from images
# =============================================================================
segmented_bounding_boxes = gpd.read_file("./data/segmented_bboxes.gpkg")

#labeled bboxes
bbox_labeled = gpd.read_file("./data/bbox_labeled_v1.gpkg")
bbox_labeled = bbox_labeled.loc[bbox_labeled.labeled==True]

#join the segmented bounding boxes with labeled
bbox_labeled = gpd.sjoin(segmented_bounding_boxes, bbox_labeled, predicate="intersects", how="inner").reset_index(drop=True)
bbox_labeled = bbox_labeled.drop("index_right", axis=1)

# =============================================================================
# Get linestrings within bbox
# =============================================================================
ls_gdf = gpd.read_file("./data/data_labeled_v1.gpkg")
ls_gdf = ls_gdf.loc[ls_gdf.StrandType.notna()]
ls_gdf = ls_gdf[["StrandType", "geometry"]]
test = gpd.sjoin(ls_gdf, bbox_labeled, predicate="intersects", how="inner").reset_index(drop=True)
test.crs = "EPSG:32633"

# =============================================================================
# for each linestring find nearest neighbors and sort the dataframe
# =============================================================================
from sklearn.neighbors import BallTree
tree_start = BallTree(test['geometry'].apply(lambda x: (x.coords[0])).tolist())
tree_end = BallTree(test['geometry'].apply(lambda x: (x.coords[-1])).tolist())

def find_nearest_neighbors_2(gdf, tree, k, start=True):
    if start:
        query_points = np.array(gdf['geometry'].apply(lambda x: (x.coords[-1])).tolist())
    else:
        query_points = np.array(gdf['geometry'].apply(lambda x: (x.coords[0])).tolist())
    ind = []
    dist = []
    for point in query_points:
        distances, indices = tree.query([point], k=k)
        ind.append(indices.squeeze())
        dist.append(distances.squeeze())
    return ind, dist

ind_start, dist_start = find_nearest_neighbors_2(test, tree_start, k=2)

# if the point closest to the geometry is itself choose next
for i in range(len(ind_start)):
    if(i == ind_start[i][0]):
        ind_start[i][0] = ind_start[i][1]
        dist_start[i][0] = dist_start[i][1]
        
ind_start = np.squeeze(np.array(ind_start))
ind_start = ind_start[:,0]
dist_start = np.squeeze(np.array(dist_start))
dist_start = dist_start[:,0]

ind_end, dist_end = find_nearest_neighbors_2(test, tree_end, k=2, start=False)
for i in range(len(ind_start)):
    if(i == ind_end[i][0]):
        ind_end[i][0] = ind_end[i][1]
        dist_end[i][0] = dist_end[i][1]
        
ind_end = np.squeeze(np.array(ind_end))
ind_end = ind_end[:,0]
dist_end = np.squeeze(np.array(dist_end))
dist_end = dist_end[:,0]


test['n_n_start'] = ind_start
test['dist_start'] = dist_start
test['n_n_end'] = ind_end
test['dist_end'] = dist_end
test = test.reset_index(drop=True)

test["start"] = False
test.loc[(test.dist_end > 0), "start"] = True

start = test.loc[test.start==True].index.values

unused = [i for i in range(test.shape[0])]
used = []


for i in range(len(start)):
    used.append(start[i])
    unused.remove(start[i])
    new_geom = False
    while not new_geom:
        prev = used[-1]
        new = test.iloc[prev].n_n_start
        if new in used or test.iloc[prev].dist_start > 0:
            new_geom = True
            break
        unused.remove(new)
        used.append(new)


used.append(unused[0])
unused.remove(unused[0])
while(len(unused)>0):
    prev = used[-1]
    new = test.iloc[prev].n_n_start
    while new in used:
        if new in unused:
            unused.remove(new)
        new = unused[0]
    unused.remove(new)
    used.append(new)
    
test = test.reindex(used)




test["st_sep"] = test.groupby("index_right")["StrandType"].transform(lambda st: st != st.shift(1))
test["dist"] = test.groupby("index_right")["geometry"].apply(lambda geometries: geometries.distance(geometries.shift(1))).reset_index(level=0, drop=True)
test["geometry_id"] = test.groupby("index_right").apply(lambda x: ((x.st_sep == True) | (x.dist>0))).cumsum().reset_index(level=0, drop=True)



def combine_linestrings(geometries):
    return LineString([point for geom in geometries for point in geom.coords])

aggregated_geometries = test.groupby(["index_right", "geometry_id"]).agg({'geometry': combine_linestrings,
                                                                          'StrandType': 'first',
                                                                          'mdFilename': 'first',
                                                                          'filename': 'first'}).reset_index()


#TODO make more efficient
# =============================================================================
# # finds connected geometries and checks if they are the same type, if yes assign the same id
# =============================================================================
# geom_order = test.geometry_id.unique()
# for geom_id in geom_order:
#     geom_idx = geom_id - 1 
#     tmp = aggregated_geometries.loc[geom_idx].geometry.touches(aggregated_geometries.geometry)
#     touches_idx = tmp.loc[(tmp == True) & (tmp.index != geom_idx)]
#     if touches_idx.empty:
#         continue
#     touches_idx= touches_idx.index[0]
#     geom_1 = aggregated_geometries.loc[geom_idx]
#     geom_2 = aggregated_geometries.loc[touches_idx]
#     if (aggregated_geometries.loc[geom_idx].index_right == aggregated_geometries.loc[touches_idx].index_right):
#         if (aggregated_geometries.loc[geom_idx].StrandType == aggregated_geometries.loc[touches_idx].StrandType):
#             aggregated_geometries.loc[touches_idx, 'geometry_id'] = aggregated_geometries.loc[geom_idx, 'geometry_id']
# aggregated_geometries['geometry'] = aggregated_geometries.groupby("geometry_id").apply(lambda x: x.unary_union)




aggregated_geometries = gpd.GeoDataFrame(aggregated_geometries, geometry='geometry')
aggregated_geometries.crs = "EPSG:32633"
#fix misspelling from labling
aggregated_geometries.loc[(aggregated_geometries.StrandType=="Sandtrand") | (aggregated_geometries.StrandType=="Sanstrand"), "StrandType"] = "Sandstrand"
# aggregated_geometries.to_file("./data/aggregated_v2.gpkg", driver="GPKG")
aggregated_geometries["bbox"] = aggregated_geometries.geometry.apply(lambda x: [box(*x.bounds)])

# =============================================================================
# id the different classes and generate classes.txt
# =============================================================================

classes = aggregated_geometries.StrandType.unique()
if classes[0] != "Background":
    for i in range(len(classes)):
        if classes[i] == "Background":
            classes[i] = classes[0]
            classes[0] = "Background"
            break
class_id = {}
for i in range(len(classes)):
    class_id[classes[i]] = i
    
# =============================================================================
# combine beach types into single class
# =============================================================================

for k, _ in class_id.items():
    if k.endswith("strand"):
        class_id[k] = 2

class_id["Menneskeskapt struktur"] = 3

label_path = "./data/labels"
classes_file_path = "classes.txt"
with open(os.path.join(label_path, classes_file_path), 'w') as classes_file:
    for k,v in class_id.items():
        classes_file.write(f"{v}: {k}\n")

aggregated_geometries["StrandType_id"] = aggregated_geometries.StrandType.map(class_id)


# =============================================================================
# Generate mask from geometries
# saves mask to files to fix memory issue
# =============================================================================

# prev_img = ""

# for idx, row in aggregated_geometries.iterrows():
#     if row.filename != prev_img:
#         src = rasterio.open(f'./data/test/{row.filename}')
#     mask = geometry_mask(geometries=[row.geometry.buffer(5)], transform=src.transform, invert=True, out_shape=src.shape)
#     mask = mask * row.StrandType_id
    
#     filename = os.path.splitext(row.filename)[0] + "_"
#     np.save(f"./data/masks/{filename + str(idx)}", mask)



# geometry_masks = []
def combine_masks(image, masks):
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        mask = np.flipud(mask) # for some reason these were inverted
        combined_mask = np.where(mask > 0, mask, combined_mask)
        
    plt.imshow(combined_mask)
    plt.axis('off')
    plt.show()
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    # one hot encodes the class along the 3rd dimension
    # do this prior to feeding the model instead, uses alot of memory
    # combined_mask = np.eye(4)[combined_mask]
    return combined_mask
    
    #    
    # geometry_masks.append((image_id, combined_mask))

prev_img = ""

for _, data in aggregated_geometries.groupby("filename"):
    masks = []
    for idx, row in data.iterrows():
        if row.filename != prev_img:
            src = rasterio.open(f'./data/test_2/{row.filename}')
            prev_img = row.filename
        mask = geometry_mask(geometries=[row.geometry.buffer(5)], transform=src.transform, invert=True, out_shape=src.shape)#TODO buffer size?
        # resized_mask = cv2.resize(mask.astype(np.uint8), (800, 300)) 
        # resized_mask = resized_mask * row.StrandType_id
        mask = mask * row.StrandType_id
        masks.append(mask)
    # image = src.read(1)#TODO use resized
    resized_img_path = "./data/test_2/"
    image = cv2.imread(resized_img_path + prev_img)
    filename = os.path.splitext(row.filename)[0]
    combined_mask = combine_masks(image, masks)
    np.save(f"./data/labels/train_v2/{filename}", combined_mask)




for image_id, data in aggregated_geometries.groupby("filename"):
    with rasterio.open(f'./data/test/{image_id}') as src:
            image = src.read(1)
    combine_masks(f"resized_640_{image_id}", image, data.geom_mask)
    
# =============================================================================
# Generate geometries from mask
# =============================================================================
image_dir = "./data/images/norge_i_bilder_2/"
geometries = []
for _, data in aggregated_geometries.iterrows():
    
    with rasterio.open(image_dir+data.mdFilename) as src:
        # Load the mask image
        coordinates = []
        for mask in data.geom_mask:
            #scale up the masks to their original shape
            original_resolution = (8000, 6000)
            current_resolution = (640, 480)
            scale_factor = (original_resolution[0] / current_resolution[0], original_resolution[1] / current_resolution[1])
            scaled_up_mask = cv2.resize(mask, original_resolution, interpolation=cv2.INTER_LINEAR)

        
            contours, _ = cv2.findContours(scaled_up_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                for pixel in contours:
                    coordinates.append(src.xy(pixel[0][0][0], pixel[0][0][1]))
                    
                    
        linestring = LineString(coordinates)
        geometries.append(linestring)
    break
        # Create a GeoDataFrame from the geometries
gdf = gpd.GeoDataFrame({'geometry': geometries})
gdf.crs = "EPSG:32633"
gdf.to_file("./data/geometries_from_mask.gpkg", driver="GPKG")






