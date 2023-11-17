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
from yoloDetector import YOLODetector , YoloThread

def renderAllDetections(frame, yres, scale):
    
     
    height = frame.shape[0]
    width = frame.shape[1]
    renderFrame = frame.copy()
    #render all
    for obj in yres:
        # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
        (x, y, x2, y2) = obj.bbox
        if obj.class_id >= 0:
            cv2.rectangle(renderFrame, (x, y), (x2, y2), (255, 0, 0), 4)
            if obj.segmentation is not None:
                cv2.polylines(renderFrame, [obj.segmentation], True, (0, 0, 255), 4)

            cv2.putText(renderFrame,   'Tree :' + str(obj.score), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    renderFrame = cv2.resize(renderFrame, (width // scale, height // scale))

    cv2.imshow("great_frame", renderFrame)
    cv2.waitKey(10)

    return renderFrame


ys = None
resource_path = 'e:/Resources/Municipio/Drone/'
working_path = 'D:/Proyects/Municipio/TreeDetectionTandil/'

################################################
def detectUsingYolo(frame):
    global ys
    # Segmentation detector
    if ys is None:
        ys = YOLODetector(working_path +'/model/yolov8_trees_17nov2023.pt')
        
    height = frame.shape[0]
    width = frame.shape[1]
    ys.detect(frame)

    yres = []

    renderF = frame.copy()

    if len(ys.objects)>0:
        ys.drawOwnDetections(renderF)

        cv2.imshow("frame_with_detections", renderF)

        cv2.putText(renderC, f"{index}",mp,
                                cv2.FONT_HERSHEY_COMPLEX, 1.2, (250, 220, 255), 2, cv2.LINE_AA)


        renderAllDetections(frame, ys.objects, 10)

        cv2.waitKey(100)

    return len(ys.objects)
       


#open point shapefile

pointData = gpd.read_file(resource_path + '/drone_explore_area.geojson')
print('CRS of Point Data: ' + str(pointData.crs))


# Opening JSON file
f = open(resource_path + '/drone_explore_area.geojson')
geojson = json.load(f)

#open raster file
tandilRaster = rasterio.open(resource_path + 'zona 2/odm_orthophoto/odm2.tif')
print('CRS of Raster Data: ' + str(tandilRaster.crs))
print('Number of Raster Bands: ' + str(tandilRaster.count))
print('Interpretation of Raster Bands: ' + str(tandilRaster.colorinterp))

#######################################################
## Load the image and split into clips
crop_width = 1000
crop_height = 1000

clipper = ImageForClipModel(image_address=resource_path + 'zona 2/odm_orthophoto/odm2.tif')
clipper.clip_raster(height=crop_width,
    width=crop_height,
    buffer=0,
    save_mode=False,
    prefix='raster_clip/clipped_band_',
    pass_empty=False)

### for each clip
index = 0
for clip in clipper.image_data:
    
    renderC = clip['crop']
    mp = (int(50), int(50))
    
    number_of_black_pix = np.sum(renderC == 0) / 3

    if ((number_of_black_pix*1.0) / (crop_height*crop_width)) < 0.75:
  #  cv2.imshow('clip', renderC)
        detections = detectUsingYolo(renderC)

        if detections > 5:
            cv2.imwrite(working_path +'assets/img_'+str(index)+'.jpg', renderC )

    
    index += 1

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