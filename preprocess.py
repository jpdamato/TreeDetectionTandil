import cv2
import numpy as np
import json
import pandas as pd
import geopandas as gpd
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_image
from functools import partial
from rasterio.enums import MergeAlg
from functools import partial
import rasterio


geo_transform = None

    
def exportDetections(path_out, detections, outputCrs = "urn:ogc:def:crs:OGC:1.3:CRS84"  ):
    header = {}
    header["type"] = "FeatureCollection"
    header["name"]: "drone_explore_area"
    header["crs"] = {"type": "name", "properties": {"name":outputCrs }}
    header["features"] = detections

    # Serializing json
    json_serialized = json.dumps(header, indent=4)

    # Writing to sample.json
    with open(path_out +"detections.json", "w") as outfile:
        outfile.write(json_serialized)
########################################################
def getLatLongBoundingBox(bbox, src_data):
    base_transform = src_data['transform']
    (x, y, x2, y2) = bbox
   # X , Y
    pixelsCoords = [[y,x],  [y,x2],
                    [y2,x2],   [y2,x],      [y,x]]
    latLongCoordinates = []
    
    for px in pixelsCoords:
        xs, ys = rasterio.transform.xy(base_transform, [px[0]], [px[1]])
        lons= np.array(xs)
        lats = np.array(ys)
        latLongCoordinates.append([lons[0], lats[0]])
    return latLongCoordinates

    
def generateDetectionAsFeature(detection, src_data, id):
    # Create a Window and calculate the transform from the source dataset    
    radio = 5

    if detection.geoBBox is None:
        detection.geoBBox = getLatLongBoundingBox(detection.bbox, src_data)
    properties = {}
    properties["classname"] = detection.className
    properties["score"] = detection.score
    properties["id"] = id
    properties["height"] = 0
    properties["radius"] = 1
    
    geometry = {}
    geometry["type"] = "MultiLineString"
    geometry["coordinates"] =[  detection.geoBBox ]
   
    detection = {}
    detection["type"] = "Feature"
    detection["properties"] = properties
    detection["geometry"] = geometry

   # Serializing json
    #json_serialized = json.dumps(detection, indent=4)

    return detection

   # {"type": "Feature",
   # "properties": {"id": 0.0, "Nombre": "de Liniers santiago", "Tipo": "Calle", "de izquier": 400.0, "a izquierd": 498.0, "de derecha": 401.0, "a derecha": 499.0, "Observació": null},
   # "geometry": { "type": "MultiLineString", "coordinates": [ [ [ -59.110606739636815, -37.320052752793728 ], [ -59.111747602041255, -37.319145463269841 ] ] ] } },

def selectPolygonsByROI(refPolygon, bounding_box, saveIntermediate, tempDir):

    # Create a custom polygon
    #polygon = box(-87.8, 41.90, -87.5, 42)
   # bounding_box = refImage.envelope
    print (refPolygon.bounds)
    clipped = refPolygon.clip(bounding_box)

    if saveIntermediate:
        clipped.to_file(tempDir + "clipped_vial.geojson", driver="GeoJSON")

    return clipped
   

#################################################
def polygonsToRaster(streets, saveIntermediate, path):
    
    # Using GeoCube to rasterize the Vector
    rastered = make_geocube(
        vector_data=streets,
        resolution=(-0.00001, 0.00001),       
        fill=0,
        rasterize_function=partial(rasterize_image, all_touched=True),
    )
    
    # Save raster census raster
    if saveIntermediate:
        rastered.rio.to_raster(path + '/streets_raster.tiff')
    return rastered

#########################################################################
## link = https://stackoverflow.com/questions/65530597/geopandas-rasterio-isolate-a-vector-as-png
def generateStreetMask(src_image, saveIntermediate, path):
    
     # reload image and extract features
    out_transform = None

    image = cv2.imread(src_image, )
    with rasterio.open(src_image, 'r') as src:
            out_transform = src.transform
            out_meta = src.meta
   
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of hue and saturation for gray colors
    # These values are approximate and can be adjusted based on your specific needs
    lower_gray = np.array([0, 0, 140], dtype=np.uint8)
    upper_gray = np.array([179, 20, 220], dtype=np.uint8)

    # Create a mask to select gray pixels
    gray_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)

    # Apply erosion filter
    kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
    eroded_mask = cv2.erode(gray_mask, kernel, iterations=1)

    # Apply dilation filter
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=dilated_mask)

    height = image.shape[0]
    width = image.shape[1]
    scale = 4

    renderFrame = cv2.resize(result_image, (width // scale, height // scale))

    # Display the result or save it
    cv2.imshow('Gray Pixels Only', renderFrame)
    cv2.waitKey(0)

    out_meta.update({"driver": "GTiff",
            "height": result_image.shape[1],
            "width": result_image.shape[2],
            "count" : 4,
            "nodata": 255,
            "transform": out_transform})

    if saveIntermediate:       

        with rasterio.open(path + "/streets_detected.tiff", 'w', **out_meta) as dst:
            dst.write(result_image , 1)