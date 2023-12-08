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
import preprocess as Processor
import os.path
from shapely.geometry import Polygon, LineString, Point

import toolkit as Toolkit
import yoloDetector as Detector
import preprocess as preProcessor

ys = None
# calculate Width from geo-coordinates
def computeTreeWidth(d):
    
    if d.geoBBox is None:
        return 0

    p0 = d.geoBBox[0]
    p1 = d.geoBBox[0]

    return min(abs(p0[0] - p1[0]), (p1[1]-p1[0])) 

# take an area around tree, and return the MAX-MIN
def computeTreeHeight(d, dem_image):
    (x, y, x2, y2) = d.bbox
    cropped_dem =  dem_image[y:y2,    x: x2]
                                
    cv2.imshow("dem", cropped_dem)

    cv2.waitKey(0)
                                
    return 0

################################################
def drawDetections(frame, objects):
        if objects is not None:
            for obj in objects:
            # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
                (x, y, x2, y2) = obj.bbox
                if obj.class_id >= 0:
                    cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
                    if obj.segmentation is not None:
                        cv2.polylines(frame, [obj.segmentation], True, (0, 0, 255), 4)

                    cv2.putText(frame, obj.className+ ':' + str(obj.score), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
 
#####################################################################
def detectUsingYolo(working_path,frame , geo_data, postProcess):
    global ys
    # Segmentation detector
    if ys is None:
        ys = Detector.YOLODetector(working_path +'/model/yolov8_trees_17nov2023.pt')
        
    height = frame.shape[0]
    width = frame.shape[1]
    ys.detect(frame,postProcess)

    yres = []

    renderF = frame.copy()

    if len(ys.objects) > 0:
        for obj in ys.objects:
            obj.geoBBox = Processor.getLatLongBoundingBox(obj.bbox,geo_data)
        drawDetections(renderF,ys.objects)
       
        cv2.imshow("frame_with_detections", renderF)

        cv2.waitKey(100)

    return ys.objects


###########################################################################3
def cropAndProcess(working_path, all_features, src_image_path , dem_image_path, cropSize=1000, red_vial = None):
    
    print ("---------------------------------------- ")
    print ("Start processing IMAGE ")
    print ("---------------------------------------- ")
    
    
    #######################################################
    ## Load the image and split into clips
    crop_width = cropSize
    crop_height = cropSize

    ## split image into sections
    clipper = Toolkit.ImageForClipModel(image_address=src_image_path)
    clipper.clip_raster(height=crop_width,
        width=crop_height,
        buffer=0,
        save_mode=False,
        prefix=working_path + '/clipped_band_',
        pass_empty=False)

    
# Get dem 
    if os.path.isfile(dem_image_path):
        
        dem_imagePIL = Image.open(dem_image_path)
        #dem_image = cv2.imread(dem_image_path,  flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
     #   dem_image.dtype = np.float
        dim = (clipper.bands.shape[1], clipper.bands.shape[0])
        dem_image = np.array(dem_imagePIL)
       
        dem_image = cv2.resize(dem_image, dim,  interpolation=cv2.INTER_AREA)

    totalClips = len(clipper.image_data)
    index = 0
    for clip in clipper.image_data:
        renderC = clip['crop']
        mp = (int(50), int(50))
        number_of_black_pix = np.sum(renderC == 0) / 3

        index += 1

        if (index % 1000 == 0):
            print ("Progress", (100*index)/totalClips)

        if ((number_of_black_pix*1.0) / (crop_height*crop_width)) < 0.75:
        ###########################################3
            bbox = [0, 0, crop_width, crop_height]
            # Uncomment for checking blocks
           # d = DetectedObject()
           # d.bbox = bbox
           # md = Processor.generateDetectionAsFeature(d , clip, len(all_features))
           # all_features.append(md)
            detections = detectUsingYolo(working_path,renderC,clip, True)

            for d in detections:
                md = Processor.generateDetectionAsFeature(d, clip, len(all_features))
                
                tree_height = computeTreeHeight(d, dem_image)
                md['properties']['tree_height'] = tree_height

                tree_width = computeTreeWidth(d)
                md['properties']['tree_width'] = tree_width

                if red_vial is not None:
                    p = d.geoBBox[0] 
                # Calculating distance from the Point to ALL Polygons
                    red_vial['distances'] = red_vial.distance(Point(p[0],p[1]) )
                # Subsetting to keep only the 3 nearest cases
                    polypdproj_subset = (red_vial.loc[red_vial['distances']
                                   .rank(method='first', ascending=True) <= 1]
                     .sort_values(by='distances', ascending=True))
                     
                    
                    
                    md['properties']['distance_to_street'] = polypdproj_subset.distances.values[0]
                    md['properties']['street_center.x'] = polypdproj_subset.centroid.values[0].x
                    md['properties']['street_center.y'] = polypdproj_subset.centroid.values[0].y
                    md['properties']['street'] = polypdproj_subset.Nombre.values[0]

                all_features.append(md)

            #if len(detections) > 5:
                ##cv2.imwrite(working_path + 'assets/img_' + str(index) + '.jpg', renderC)            
            #    Processor.exportDetections(tempPath,all_features,tandilRaster.crs.wkt)

  