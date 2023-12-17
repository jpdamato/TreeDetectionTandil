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
MINIMAL_EXPECTED_HEIGHT = 50


def calcul_area(box):
    x1, y1, x2, y2 = box
    return abs(x1 - x2) * abs(y1 - y2)

def nms_area(_box, boxes, thresh_iou: float) -> float:
    
    max_iou = 0.0

    box_lhs = _box['properties']['bbox']

    for box in boxes:

        box_rhs = box['properties']['bbox']
    
        x1_lhs, y1_lhs, x2_lhs, y2_lhs = box_lhs
        x1_rhs, y1_rhs, x2_rhs, y2_rhs = box_rhs

        area_lhs = calcul_area(box_lhs)
        area_rhs = calcul_area(box_rhs)

        # Determines the coordinates of the intersection box
        x1_inter = max(x1_lhs, x1_rhs)
        y1_inter = max(y1_lhs, y1_rhs)
        x2_inter = min(x2_lhs, x2_rhs)
        y2_inter = min(y2_lhs, y2_rhs)

        # Determines if the boxes overlap or not
        # If one of the two is equal to 0, the boxes do not overlap
        inter_w = max(0.0, x2_inter - x1_inter)
        inter_h = max(0.0, y2_inter - y1_inter)

        if inter_w == 0.0 or inter_h == 0.0:
            continue

        intersection_area = inter_w * inter_h
        union_area = area_lhs + area_rhs - intersection_area

        # See if the smallest box is not mostly in the largest one
        if intersection_area / area_rhs >= thresh_iou:
            iou = area_rhs / intersection_area
        else:
            iou = intersection_area / union_area

        max_iou = max(iou, max_iou)

    return max_iou


# calculate Width from geo-coordinates
def computeTreeWidth(d):
    
    if d.geoBBox is None:
        return 0

    p0 = d.geoBBox[0]
    p1 = d.geoBBox[0]

    return min(abs(p0[0] - p1[0]), (p1[1]-p1[0])) 

def computeGreenAmount(d, green_band):
    (x, y, x2, y2) = d.bbox
    # choose all image
    cropped_green = green_band[y:y2, x:x2, 1]
    cropped_red = green_band[y:y2, x:x2, 0]

    if (cropped_green.shape[1] * cropped_green.shape[0]) == 0:
        return 0
    
    # choose only center
    if ((y+y2)//2 > green_band.shape[0]) or ((x+x2)//2 > green_band.shape[0]):
        return 0
    
   
    filter_green = cropped_green[cropped_green > 1]
    filter_red = cropped_red[cropped_red > 1]

    coefG = len(filter_green) / ((y2 - y) * (x2 - x))
    coefR = len(filter_red)/((y2-y)*(x2-x))
   
    if (coefR > 0.4):
        cv2.imshow("greenC", cropped_red)
        cv2.waitKey(1)
    
    return coefG 


# take an area around tree, and return the MAX-MIN
def computeTreeHeight(d, dem_image):
    (x, y, x2, y2) = d.bbox
    # choose all image
    cropped_dem = dem_image[y:y2, x:x2]

    if (cropped_dem.shape[1] * cropped_dem.shape[0]) == 0:
        return 0
    
    # choose only center
    if ((y+y2)//2 > dem_image.shape[0]) or ((x+x2)//2 > dem_image.shape[0]):
        return 0

    meanH = dem_image[(y+y2)//2 ,(x+x2)//2 ]
    
    # BUG Here.. asuming minimal height of pixel
    filter_arr = cropped_dem[cropped_dem > MINIMAL_EXPECTED_HEIGHT]
    
    if cropped_dem.max() > 0:
        cropped_dem = cv2.normalize(cropped_dem, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        colormapped_image = cv2.applyColorMap(cropped_dem, cv2.COLORMAP_JET)        
        cv2.imshow("dem", colormapped_image)
        cv2.waitKey(1)

        if meanH > filter_arr.min():
            return meanH - filter_arr.min()
        else:        
            return filter_arr.max() - filter_arr.min()
    else:
        return 0

#remove overlapped elements
def removeOverlapped(boxes1, boxes2 , discarded):

    selected_boxes = boxes1

    for b1 in boxes2:
        
        overlap = nms_area(b1, boxes1, 0.2)
        # no overlaps
        if overlap < 0.20:
            selected_boxes.append(b1)
        else:
            discarded.append(b1)

    return selected_boxes



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
def cropAndProcess(working_path, all_features, src_image_path , dem_image_path,green_path, cropSize=1000, red_vial = None):
    
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

    green_band = None
    if os.path.isfile(green_path):
        green_band = cv2.imread(green_path)

# Get dem 
    if os.path.isfile(dem_image_path):
        
        dem_imagePIL = Image.open(dem_image_path)
        #dem_image = cv2.imread(dem_image_path,  flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
     #   dem_image.dtype = np.float
        dim = (clipper.bands.shape[1], clipper.bands.shape[0])
        dem_image = np.array(dem_imagePIL)
        dem_image = np.where(dem_image < -1, -1, dem_image)

        
        dem_image = cv2.resize(dem_image, dim,  interpolation=cv2.INTER_AREA)

    totalClips = len(clipper.image_data)
    index = 0
    for clip in clipper.image_data:
        renderC = clip['crop']
        cropped_dem_image = None
        mp = (int(50), int(50))
        number_of_black_pix = np.sum(renderC == 0) / 3

        index += 1

        if (index % 1000 == 0):
            print("Progress", (100 * index) / totalClips)

        if green_band is not None:
            cropped_green_image =  green_band[clip['row_offset']:clip['row_offset'] + clip['height'],
                                clip['col_offset']: clip['col_offset']+clip['width'] ]
            cv2.imshow("green", cropped_green_image)
            
        if dem_image is not None:
            cropped_dem_image =  dem_image[clip['row_offset']:clip['row_offset'] + clip['height'],
                                clip['col_offset']: clip['col_offset']+clip['width'] ]
            cropped_dem_imageR = cv2.normalize(cropped_dem_image, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            cv2.imshow("depth", cropped_dem_imageR)
            

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
                
                green_count = computeGreenAmount(d, cropped_green_image)

                tree_height = computeTreeHeight(d, cropped_dem_image)

                md['properties']['green_count'] = green_count

                if hasattr(tree_height,"T"):
                    md['properties']['tree_height'] = float(tree_height)
                else:
                    md['properties']['tree_height'] = tree_height

                tree_width = computeTreeWidth(d)
                md['properties']['tree_width'] = tree_width

                x = min(d.geoBBox[0][0], d.geoBBox[2][0])
                y = min(d.geoBBox[0][1], d.geoBBox[2][1])
                x2 = max(d.geoBBox[0][0], d.geoBBox[2][0])
                y2 = max(d.geoBBox[0][1], d.geoBBox[2][1])
                
                md['properties']['bbox'] = [x,y,x2,y2]

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

  