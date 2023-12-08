#https://pysource.com/2023/02/21/yolo-v8-segmentation
import cv2

import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import json
import affine
from rasterio.plot import show
from rasterio.windows import Window
#from skimage.feature import match_template
import numpy as np
from PIL import Image
import os.path
from shapely.geometry import Polygon, LineString, Point

import toolkit as Toolkit
import yoloDetector as Detector
import preprocess as preProcessor
import pipeline as Pipeline

resource_path = 'e:/Resources/Municipio/Drone/'
working_path = 'D:/Proyects/Municipio/TreeDetectionTandil/'

#################################################################
## MAIN
#################################################################
#open point shapefile
# Common files
pointData = gpd.read_file(resource_path + '/drone_explore_area.geojson')
print('CRS of Point Data: ' + str(pointData.crs))

vialData = gpd.read_file(resource_path + '/red_vial_tandil.geojson')



# Opening JSON file
f = open(resource_path + '/drone_explore_area.geojson')
geojson = json.load(f)

processingZone = "zona 8"
tempPath = "E:/Resources/Municipio/Drone/temp/"+processingZone+"/"

##Start processing
#open raster file to check 
imageBounds = None
tandilRaster = rasterio.open(resource_path + processingZone + '/odm_orthophoto/odm_orthophoto.tif')

vialData = vialData.to_crs(tandilRaster.crs.to_dict())
print('CRS of Vial Data: ' + str(vialData.crs))

print('CRS of Raster Data: ' + str(tandilRaster.crs))
print('Number of Raster Bands: ' + str(tandilRaster.count))
print('Interpretation of Raster Bands: ' + str(tandilRaster.colorinterp))
imageBounds = tandilRaster.bounds

src_image_path = resource_path + processingZone +'/odm_orthophoto/odm_orthophoto.tif'
dem_image_path = resource_path + processingZone +'/odm_dem/dsm.tif'

if not os.path.exists(tempPath):
    os.mkdir(tempPath)

# extract Street mask
if not os.path.isfile(tempPath + '/streets_detected.png'):
    preProcessor.generateStreetMask( src_image, True, tempPath)

##select vial data 
if not os.path.isfile(tempPath + '/clipped_vial.geojson'):
    preProcessor.selectPolygonsByROI(vialData,imageBounds, True, tempPath)
clipped_vial = gpd.read_file(tempPath + '/clipped_vial.geojson')

# clip 
if not os.path.isfile(tempPath + '/streets_raster.png'):
    preProcessor.polygonsToRaster(clipped_vial, True, tempPath)

### for each clip, run Yolo and add as a feature

if not os.path.isfile(tempPath + '/detections_x2.json'):
    all_features = []
    Pipeline.cropAndProcess(working_path,all_features, src_image_path , dem_image_path, 2000, vialData)
    Toolkit.exportDetections(tempPath,all_features,tandilRaster.crs.wkt,"detections_x2.json" )

if not os.path.isfile(tempPath + '/detections_x1.json'):
    all_features = []
    Pipeline.cropAndProcess(working_path,all_features, src_image_path , dem_image_path, 1000, vialData)
    Toolkit.exportDetections(tempPath, all_features, tandilRaster.crs.wkt, "detections_x1.json")
    