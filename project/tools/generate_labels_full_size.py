
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as ET
import os
import numpy as np
from shapely.geometry import Polygon, box, LineString

# =============================================================================
# Get bounding box from images
# =============================================================================

def list_xml_files(folder_path):
    xml_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            xml_files.append(os.path.join(folder_path, filename))
    return xml_files

image_folder = "./data/images/norge_i_bilder_2/"
xml_files = list_xml_files(image_folder)
data = []
for xml_file in xml_files:
    with open(xml_file, 'r', encoding='utf-8') as file:
        xml_data = file.read()
    root = ET.fromstring(xml_data)
    content = {}
    for element in root:
        content[element.tag] = element.text
    data.append(content)

df = pd.DataFrame(data)
df.dropna(subset=["eastBP"], inplace=True)
# df["geometry"] = df.apply(lambda x: Polygon([(x.westBP, x.southBP), (x.westBP, x.northBP), (x.eastBP, x.northBP), (x.eastBP, x.southBP)]), axis=1)
df["geometry"] = df.apply(lambda x: box(x.westBP, x.southBP, x.eastBP, x.northBP), axis=1)
# df["labeled"] = False

gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.crs = "EPSG:32633"
# gdf.to_file("./data/image_bboxes_2.gpkg", driver="GPKG")

#labeled bboxes
bbox_labeled = gpd.read_file("./data/bbox_labeled_v1.gpkg")
bbox_labeled = bbox_labeled.loc[bbox_labeled.labeled==True]

# =============================================================================
# Get linestrings within bbox
# =============================================================================
ls_gdf = gpd.read_file("./data/data_labeled_v1.gpkg")
ls_gdf = ls_gdf.loc[ls_gdf.StrandType.notna()]
ls_gdf = ls_gdf[["StrandType", "geometry"]]
test = gpd.sjoin(ls_gdf, bbox_labeled, predicate="intersects", how="inner").reset_index(drop=True)
test.crs = "EPSG:32633"
# test.to_file("./data/test_v1.gpkg", driver="GPKG")
# test = test.loc[test.index_right.isin([10, 11])]

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


# =============================================================================
# recursively go through the generated geometries to check if geometrys are connected and have the same type
# we know the order is correct because we have used the correct starting geometries from earlier step
# assign new id if connected and equal to the id of the first geometry
# =============================================================================
test["touches"] = test.geometry.touches(test.geometry.shift(1))
# test.groupby("geometry_id").apply(lambda x: x.at[x.index[-1], "touches"] = True)



import rasterio
import cv2
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
from pyproj import Proj, transform

# =============================================================================
# cretate masks from linestrings(might need for other models later)
# =============================================================================

def combine_linestrings(geometries):
    return LineString([point for geom in geometries for point in geom.coords])

aggregated_geometries = test.groupby(["index_right", "geometry_id"]).agg({'geometry': combine_linestrings,
                                                                          'StrandType': 'first',
                                                                          'mdFilename': 'first'}).reset_index()
# aggregated_geometries = gpd.GeoDataFrame(aggregated_geometries, geometry='geometry')




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
#     TODO hvis jeg vil få dette til å funke på en kjøring må neste geometrien jeg bruker være touches_idx. da vil jeg følge riktig retning hele veien(recursive)
# aggregated_geometries['geometry'] = aggregated_geometries.groupby("geometry_id").apply(lambda x: x.unary_union)




aggregated_geometries = gpd.GeoDataFrame(aggregated_geometries, geometry='geometry')
aggregated_geometries.crs = "EPSG:32633"
#fix misspelling from labling
aggregated_geometries.loc[(aggregated_geometries.StrandType=="Sandtrand") | (aggregated_geometries.StrandType=="Sanstrand"), "StrandType"] = "Sandstrand"
# aggregated_geometries.to_file("./data/aggregated_v1.gpkg", driver="GPKG")

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

label_path = "./data/labels"
classes_file_path = "classes.txt"
with open(os.path.join(label_path, classes_file_path), 'w') as classes_file:
    for k,v in class_id.items():
        classes_file.write(f"{v}: {k}\n")

aggregated_geometries["StrandType_id"] = aggregated_geometries.StrandType.map(class_id)


# =============================================================================
# Generate mask from geometries
# =============================================================================

prev_img = ""
masks = []
for idx, row in aggregated_geometries.iterrows():
    if row.mdFilename != prev_img:
        src = rasterio.open(f'./data/images/norge_i_bilder_2/{row.mdFilename}')
    mask = geometry_mask(geometries=[row.geometry.buffer(5)], transform=src.transform, invert=True, out_shape=src.shape)
    resized_mask = cv2.resize(mask.astype(np.uint8), (640, 480)) 
    resized_mask = resized_mask * row.StrandType_id
    masks.append(resized_mask)
    # aggregated_geometries.at[idx, "test"] = resized_mask 

aggregated_geometries["geom_mask"] = masks


geometry_masks = []
def overlay_masks_on_image(image_id, image, masks):
    combined_mask = np.zeros_like(image, dtype=np.uint8)
    for mask in masks:
        combined_mask = np.where(mask > 0, mask, combined_mask)
        # result = cv2.bitwise_not(image,image,mask = mask)
        
    plt.imshow(combined_mask)
    plt.axis('off')
    plt.show()
    # one hot encodes the class along the 3rd dimension
    combined_mask = np.eye(len(classes))[combined_mask]    
    geometry_masks.append((image_id, combined_mask))


import glob
resized_images = glob.glob(os.path.join("./data/images/norge_i_bilder_2", "resized*"))

for image_id, data in aggregated_geometries.groupby("mdFilename"):
    with rasterio.open(f'./data/images/norge_i_bilder_2/resized_640_{image_id}') as src:
            image = src.read(1)
    overlay_masks_on_image(f"resized_640_{image_id}", image, data.geom_mask)
    
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



# =============================================================================
# Save masks to labels file
# =============================================================================
labels_path = "./data/labels/train_v1_1/"
for image_id, mask in geometry_masks:
    image_id, _ = os.path.splitext(image_id)
    np.save(labels_path + image_id, mask)


# =============================================================================
# generate bounding boxes per geometry
# =============================================================================
prev_img = ""
for idx, row in aggregated_geometries.iterrows():
    if row.mdFilename != prev_img:
        src = rasterio.open(f'./data/images/norge_i_bilder_2/{row.mdFilename}')
    mask = geometry_mask(geometries=row.bbox, transform=src.transform, invert=True, out_shape=src.shape)
    resized_mask = cv2.resize(mask.astype(np.uint8), (640, 480))
    contour, _ = cv2.findContours(resized_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if the geometry moves outside the bounds of the image no mask is returned
    if len(contour) == 0:
        aggregated_geometries.at[idx, "bbox"] = ()
    for c in contour:
        bbox = cv2.boundingRect(c)
        aggregated_geometries.at[idx, "bbox"] = bbox

def show_img_w_bboxes(img, bboxes):
    image_with_bboxes = img.copy()
    for bbox in bboxes: 
        if len(bbox) == 0:
            continue
        x, y, w, h = bbox
        cv2.rectangle(image_with_bboxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    plt.imshow(image_with_bboxes)
    plt.axis('off')
    plt.show()

for image_id, data in aggregated_geometries.groupby("mdFilename"):
    with rasterio.open(f'./data/images/norge_i_bilder_2/resized_640_{image_id}') as src:
            image = src.read(1)
    show_img_w_bboxes(image, data.bbox)
    

def convert_to_yolo_format(bbox, image_shape):
    x_center = (bbox[0] + bbox[2] / 2) / image_shape[1]
    y_center = (bbox[1] + bbox[3] / 2) / image_shape[0]
    width = bbox[2] / image_shape[1]
    height = bbox[3] / image_shape[0]
    return x_center, y_center, width, height


# drop geometries with no label due to moving outside of image bounds
aggregated_geometries = aggregated_geometries[aggregated_geometries.bbox != ()]



annotations_dir = "./data/yolo_v8/labels/all_labels"
images_dir = "./data/yolo_v8/images/all_images"
grouped_data = aggregated_geometries.groupby("mdFilename")



for image_id, data in grouped_data:

    image_path = os.path.join(images_dir, f"resized_640_{image_id}")
    annotation_path = os.path.join(annotations_dir, f"{os.path.splitext(f'resized_640_{image_id}')[0]}.txt")

    with rasterio.open(image_path) as src:
        image_shape = src.shape

    # Create a DataFrame with YOLO format annotations
    annotations_df = data.apply(lambda row: pd.Series((row["StrandType_id"],) + convert_to_yolo_format(row["bbox"], image_shape)), axis=1)
    annotations_df.columns = ["class_id", "x_center", "y_center", "width", "height"]
    annotations_df.class_id = annotations_df.class_id.apply(int)
    annotations_df.to_csv(annotation_path, header=False, index=False, sep=" ", float_format="%.6f")
    print(f"Annotations saved to {annotation_path}")
    
# =============================================================================
# Alternative to above
# =============================================================================
# for image_id, data in aggregated_geometries.groupby("mdFilename"):
#     annotation_filename = os.path.join("./data/images", f"{os.path.splitext(image_id)[0]}.txt")
#     print(os.path.splitext(image_id))
#     with rasterio.open(f'./data/images/{image_id}') as src:
#         image = src.read(1)
#     with open(annotation_filename, "w") as annotation_file:
#         for row in data[["bbox", "StrandType"]].itertuples(index=False):
#             bbox, class_id = row
#             # Calculate YOLO format coordinates (normalized values)
#             x_center = (bbox[0] + bbox[2] / 2) / image.shape[1]
#             y_center = (bbox[1] + bbox[3] / 2) / image.shape[0]
#             width = bbox[2] / image.shape[1]
#             height = bbox[3] / image.shape[0]

#             # YOLO format: class_label x_center y_center width height
#             annotation_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
#             annotation_file.write(annotation_line)
            



# =============================================================================
# generate bounding boxes for all geometries within the image
# results in memory issues
# =============================================================================
# test = test.loc[test.index_right==3]
# for _, frame in aggregated_geometries.groupby("index_right"):
    
#     with rasterio.open(f'./data/images/{frame.mdFilename.iloc[0]}') as src:
#         img = src.read(1)
#     masks = [geometry_mask(geometries=geometry, transform=src.transform, invert=True, out_shape=src.shape) for geometry in aggregated_geometries.bbox]
#     contours = np.array([contour for mask in masks for contour, _ in [cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)]])
#     contours = tuple(contours.squeeze())
#     bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    
#     image_with_bboxes = img.copy()
#     for bbox in bounding_boxes:
#         x, y, w, h = bbox
#         cv2.rectangle(image_with_bboxes, (x, y), (x + w, y + h), (0, 0, 255), 100)

#     plt.imshow(image_with_bboxes)
#     plt.axis('off')
#     plt.show()
#     break







