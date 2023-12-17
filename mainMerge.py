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
#open point shapefile
# Common files
def main(argv):
    global version, resource_path
    try:
        opts, args = getopt.getopt(argv, "hi:o:", [ "dir="])
        print(args)
        for opt, arg in opts:
            if opt in ("-d", "--dir"):
                resource_path = arg
                print("Selected directory : " + resource_path)

        ###Merge all features
        tandilRaster = None
        selected = []
        discarded = []

        ##### 
        for i in range(1, 8):
            processingZone = f"Zona {i}"
            mergedFilePath = resource_path+ "/temp/"  + processingZone + "/merged.json"
            if os.path.exists(mergedFilePath):
                print("Start merging " + f"Zona {i}")
                features = Toolkit.readDetections(mergedFilePath)
                
                if (len(selected) == 0):
                    tandilRaster = rasterio.open(resource_path + processingZone + '/odm_orthophoto/odm_orthophoto.tif')
        
                    selected = features['features']
                    
                else:
                    selected = Pipeline.removeOverlapped(selected, features['features'],discarded)
        
        ### get discarded
        for i in range(1, 8):
            processingZone = f"Zona {i}"
            mergedFilePath = resource_path+ "/temp/"  + processingZone + "/discarded.json"
            if os.path.exists(mergedFilePath):                
                discarded_features = Toolkit.readDetections(mergedFilePath)
                for tree in discarded_features['features']:
                    discarded.append(tree)
            


        if tandilRaster is not None:
            Toolkit.exportDetections(resource_path, selected, tandilRaster.crs.wkt, f"tree_layer{version}.json")

            Toolkit.exportDetections(resource_path, discarded, tandilRaster.crs.wkt, f"tree_layer{version}_discarded.json")
            print("Processed finish Ok. File saved at" + resource_path + f"tree_layer{version}.json")
    except:
        print("Error at processing. Now EXIT")
        
       
########################################
if __name__ == "__main__":
    print("########################################")    
    print("Merge all areas")
    print("Pladema version " + version)
    main(sys.argv[1:])