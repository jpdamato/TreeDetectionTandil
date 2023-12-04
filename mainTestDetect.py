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
from toolkits import ImageForClipModel
from yoloDetector import YOLODetector, YoloThread, DetectedObject
import preprocess as Processor
import os.path

ys = None
resource_path = 'e:/Resources/Municipio/Drone/'
working_path = 'D:/Proyects/Municipio/TreeDetectionTandil/'

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

def detectUsingYolo(frame , geo_data):
    global ys
    # Segmentation detector
    if ys is None:
        ys = YOLODetector(working_path +'/model/yolov8_trees_17nov2023.pt')
        
    height = frame.shape[0]
    width = frame.shape[1]
    ys.detect(frame)

    yres = []

    renderF = frame.copy()

    if len(ys.objects) > 0:
        for obj in ys.objects:
            obj.geoBBox = Processor.getLatLongBoundingBox(obj.bbox,geo_data)
        drawDetections(renderF,ys.objects)
       
        cv2.imshow("frame_with_detections", renderF)

        cv2.waitKey(100)

    return ys.objects
       


#open point shapefile

tempPath = "E:/Resources/Municipio/Drone/temp/"

pointData = gpd.read_file(resource_path + '/drone_explore_area.geojson')
print('CRS of Point Data: ' + str(pointData.crs))


vialData = gpd.read_file(resource_path + '/red_vial_tandil.geojson')
print('CRS of Vial Data: ' + str(vialData.crs))


# Opening JSON file
f = open(resource_path + '/drone_explore_area.geojson')
geojson = json.load(f)

#open raster file
tandilRaster = rasterio.open(resource_path + 'zona 2/odm_orthophoto/odm_orthophoto.tif')
print('CRS of Raster Data: ' + str(tandilRaster.crs))
print('Number of Raster Bands: ' + str(tandilRaster.count))
print('Interpretation of Raster Bands: ' + str(tandilRaster.colorinterp))


vialData = vialData.to_crs(tandilRaster.crs.to_dict())

src_image = resource_path + 'zona 2/odm_orthophoto/odm_orthophoto.tif'

dem_image = resource_path + 'zona 2/odm_dem/dsm.tif'

# extract Street mask
if not os.path.isfile(tempPath + '/streets_detected.tiff'):
    Processor.generateStreetMask( src_image, True, tempPath)

#######################################################
## Load the image and split into clips
crop_width = 1000
crop_height = 1000

## split image into sections
clipper = ImageForClipModel(image_address=src_image)
clipper.clip_raster(height=crop_width,
    width=crop_height,
    buffer=0,
    save_mode=False,
    prefix=tempPath + '/clipped_band_',
    pass_empty=False)

# split image into sections
clipperDepth = ImageForClipModel(image_address=src_image)
clipperDepth.clip_raster(height=crop_width,
    width=crop_height, demImageURL=dem_image,
    buffer=0,
    save_mode=False,
    prefix=tempPath + '/clipped_band_',
    pass_empty=False)


##select vial data 
if not os.path.isfile(tempPath + '/clipped_vial.geojson'):
    Processor.selectPolygonsByROI(vialData,clipper.bounds, True, tempPath)
clipped_vial = gpd.read_file(tempPath + '/clipped_vial.geojson')

# clip 
if not os.path.isfile(tempPath + '/streets_raster.tiff'):
    Processor.polygonsToRaster(clipped_vial, True, tempPath)

### for each clip, run Yolo and add as a feature
index = 0


all_features = []
for clip in clipper.image_data:
   # uncomment for test 
   # if len(all_features) > 1000:
   #     break

    renderC = clip['crop']
    mp = (int(50), int(50))
    
    number_of_black_pix = np.sum(renderC == 0) / 3

    if ((number_of_black_pix*1.0) / (crop_height*crop_width)) < 0.75:
  
    ###########################################3
        bbox = [0, 0, crop_width, crop_height]
        d = DetectedObject()
        d.bbox = bbox
        md = Processor.generateDetectionAsFeature(d , clip, len(all_features))
        all_features.append(md)
        
  
  #  cv2.imshow('clip', renderC)
        detections = detectUsingYolo(renderC,clip)

        for d in detections:
            md = Processor.generateDetectionAsFeature(d , clip, len(all_features))
            all_features.append(md)

        if len(detections) > 5:
            ##cv2.imwrite(working_path + 'assets/img_' + str(index) + '.jpg', renderC)            
            Processor.exportDetections(tempPath,all_features,tandilRaster.crs.wkt)

    
    index += 1

Processor.exportDetections(tempPath,all_features,tandilRaster.crs.wkt)

######################################################
#######################################################
#selected band: green
greenBand = tandilRaster.read(2)
#extract point value from raster
surveyRowCol = []

radio = 100

for index in range(0, 50):
    geometry = geojson['features'][index]['geometry']['coordinates'][0]
    ## starting line
    x = geometry[0][0]
    y = geometry[0][1]
   
    row, col = tandilRaster.index(x, y)
    
    if row < 0 or col < 0:
        continue
    
    # Create a Window and calculate the transform from the source dataset    
    window = Window(row, col, radio, radio)
    transform = tandilRaster.window_transform(window)

    # Create a new cropped raster to write to
    profile = tandilRaster.profile
    profile.update({
        'height': radio,
        'width': radio,
        'transform': transform})

    print("Point N°:%d corresponds to row, col: %d, %d"%(index,row,col))
    surveyRowCol.append([row,col])

# number of template images
print('Number of template images: %d'%len(surveyRowCol))
# define ratio of analysis

#show all the points of interest, please be careful to have a complete image, otherwise the model wont run
fig, ax = plt.subplots(1, len(surveyRowCol),figsize=(20,5))

for index, item in enumerate(surveyRowCol):
    row = item[0]
    col = item[1]
    ax[index].imshow(palmRaster)
    ax[index].plot(col,row,color='red', linestyle='dashed', marker='+',
     markerfacecolor='blue', markersize=8)
    ax[index].set_xlim(col-radio,col+radio)
    ax[index].set_ylim(row-radio,row+radio)
    ax[index].axis('off')
    ax[index].set_title(index)

#show point and raster on a matplotlib plot
#fig, ax = plt.subplots(figsize=(18,18))
#show(palmRaster, ax=ax)
#pointData.plot(ax=ax, color='orangered', markersize=100)

plt.show()

cv2.waitKey(0)