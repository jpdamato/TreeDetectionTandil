#https://pysource.com/2023/02/21/yolo-v8-segmentation
import cv2
import sys, getopt
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


version = "17Dic2023"
resource_path = 'e:/Resources/Municipio/Drone/'

#################################################################
## MAIN
#################################################################
#open point shapefile
# Common files
def main(argv):
    global version, resource_path
    try:
        processingZone = "zona 8"

        opts, args = getopt.getopt(argv, "hi:o:", ["zone=", "dir="])
        print(args)
        for opt, arg in opts:
            if opt in ("-d", "--dir"):
                resource_path = arg
                print("Selected directory : " + resource_path)
                
            if opt in ("-z", "--zone"):
                processingZone = arg
                print ("Selected zone : " + processingZone)
        
        pointData = gpd.read_file(resource_path + '/drone_explore_area.geojson')
        print('CRS of Point Data: ' + str(pointData.crs))

        vialData = gpd.read_file(resource_path + '/red_vial_tandil.geojson')

        # Opening JSON file
        f = open(resource_path + '/drone_explore_area.geojson')
        geojson = json.load(f)

        
        tempPath = resource_path + "/temp/" + processingZone + "/"
        print("Using path :" + tempPath)

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

        print("Start processing ")
        src_image_path = resource_path + processingZone +'/odm_orthophoto/odm_orthophoto.tif'
        dem_image_path = resource_path + processingZone +'/odm_dem/dsm.tif'

        #
        if not os.path.exists(tempPath):
            print ("making path " + tempPath)
            os.mkdir(tempPath)

        # extract Street mask
        if not os.path.isfile(tempPath + '/streets_detected.png'):
            preProcessor.generateStreetMask( src_image_path, True, tempPath)


        #  green band
        src_green_image = tempPath + '/greenBand.png'
        if not os.path.isfile(tempPath + '/greenBand.png'):
            preProcessor.getGreenBand(src_image_path,tempPath + '/greenBand.png')

        ##select vial data 
        if not os.path.isfile(tempPath + '/clipped_vial.geojson'):
            preProcessor.selectPolygonsByROI(vialData,imageBounds, True, tempPath)
        
        print("Reading and clipping vial ")    
        clipped_vial = gpd.read_file(tempPath + '/clipped_vial.geojson')

        # clip 
        if not os.path.isfile(tempPath + '/streets_raster.png'):
            preProcessor.polygonsToRaster(clipped_vial, True, tempPath)

        print("Preparing detections ")  
        ### for each clip, run Yolo and add as a feature
        all_features3 = []
        all_features2 = []

        if not os.path.isfile(tempPath + '/detections_x3.json'):
            print("Preparing processing Level3 ")  
            Pipeline.cropAndProcess(tempPath,all_features3, src_image_path , dem_image_path,src_green_image, 3000, vialData)
            Toolkit.exportDetections(tempPath,all_features3,tandilRaster.crs.wkt,"detections_x3.json" )

        if not os.path.isfile(tempPath + '/detections_x2.json'):
            Pipeline.cropAndProcess(tempPath,all_features2, src_image_path , dem_image_path,src_green_image, 2000, vialData)
            Toolkit.exportDetections(tempPath,all_features2,tandilRaster.crs.wkt,"detections_x2.json" )

        if not os.path.isfile(tempPath + '/detections_x1.json'):
            all_features = []
            Pipeline.cropAndProcess(tempPath,all_features, src_image_path , dem_image_path,src_green_image, 1000, vialData)
            Toolkit.exportDetections(tempPath, all_features, tandilRaster.crs.wkt, "detections_x1.json")
            
        ###Merge all features
        all_features3 = Toolkit.readDetections(tempPath + '/detections_x3.json')
        all_features2 = Toolkit.readDetections(tempPath + '/detections_x2.json')
        all_features1 = Toolkit.readDetections(tempPath + '/detections_x1.json')

        discarded = []

        selected = Pipeline.removeOverlapped(all_features3['features'], all_features2['features'],discarded)
        selected = Pipeline.removeOverlapped(selected, all_features1['features'],discarded)
        ## filter 
        filtered = []
        for tree in selected:
            if tree['properties']['distance_to_street'] < 15 and  tree['properties']['tree_height'] > 2 :
                filtered.append(tree)
            else:
                discarded.append(tree)

        Toolkit.exportDetections(tempPath, filtered, tandilRaster.crs.wkt, "merged.json")
        Toolkit.exportDetections(tempPath, discarded, tandilRaster.crs.wkt, "discarded.json")
        print("Processed finish Ok. Zone file saved at" + tempPath +  "merged.json")
    except:
        print("Error at processing. Now EXIT")

########################################
if __name__ == "__main__":
    print("########################################")    
    print("Tree detection from drone images")
    print("Pladema version 17dic2023")
    main(sys.argv[1:])