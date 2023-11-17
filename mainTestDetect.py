#https://pysource.com/2023/02/21/yolo-v8-segmentation
import cv2
from toolkit.yolo_segmentation import YOLOSegmentation , YoloThread
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



class ImageForClipModel:
    """Class for clipping big raster images into smaller parts. Class initialized with image address.
    Class has only one method with two modes of work:
    
    1 - clipped data is saved,
    2 - clipped data is stored in the list clipped_images.
    
    Default mode is (2). It is recommend to use it only during the development and debugging with small images.
    """

    # Get band, band shape, dtype, crs and transform values
    def __init__(self, image_address):
        
        self.bands = cv2.imread(image_address)

        self.snapshot = cv2.resize(self.bands, None, fx=0.10, fy=0.10, interpolation=cv2.INTER_AREA)

       
        with rasterio.open(image_address, 'r') as f:
            self.band = f.read(1)
            self.crs = f.crs
            self.base_transform = f.transform
        self.band_shape = self.band.shape
        self.band_dtype = self.band.dtype
        self.clipped_images = []
        self.clipped_addresses = []
        self.image_data = []

    # Function for clipping band
    def clip_raster(self, height, width, buffer=0, save_mode=False,
                    prefix='clipped_band_', pass_empty=False):
       
        row_position = 0
        while row_position < self.band_shape[0]:
            col_position = 0
            while col_position < self.band_shape[1]:
                cropped_image =  self.bands[row_position:row_position + height,
                                col_position:col_position + width]
               # clipped_image = self.band[row_position:row_position + height,
                #                col_position:col_position + width]

                # Check if frame is empty
                if pass_empty:
                    if np.mean(cropped_image) == 0:
                        print('Empty frame, not saved')
                        break

                # Positioning
                tcol, trow = self.base_transform * (col_position, row_position)
                new_transform = affine.Affine(self.base_transform[0], self.base_transform[1], tcol,
                                              self.base_transform[3], self.base_transform[4], trow)
                xs, ys = rasterio.transform.xy(self.base_transform, [trow], [tcol])
                lons= np.array(xs)
                lats = np.array(ys)
                
                image = {'crop':cropped_image, 'crs':self.crs, 'transform':new_transform,
                         'width':cropped_image.shape[0], 'height':cropped_image.shape[1],
                         'band':self.band_dtype , 'long':lons[0], 'lat':lats[0]}

                
                # Save or append into a set
                if save_mode:
                    filename = prefix + 'x_' + str(col_position) + '_y_' + str(row_position) + '.tif'
                    with rio.open(filename, 'w', driver='GTiff', height=image[3],
                                  width=image[4], count=1, dtype=image[5],
                                  crs=image[1], transform=image[2]) as dst:
                        dst.write(image[0], 1)
                    self.clipped_addresses.append(filename)
                else:
                    self.clipped_images.append(cropped_image)
                    self.image_data.append(image)

                # Update column position
                col_position = col_position + width - buffer

            # Update row position
            row_position = row_position + height - buffer

        if save_mode:
            print('Tiles saved successfully')
            return self.clipped_addresses
        else:
            print('Tiles prepared successfully')
            return self.clipped_images

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

def segmentateUsingYolo():

    # Segmentation detector
    #ys = YOLOSegmentation("D:/Resources/models/yolov8/yolov8n.pt")
    #videoCap =cv2.VideoCapture( "d:/Resources/Novathena/vlc-record-2023-04-20_v1.mp4")

    ys = YOLOSegmentation("e:/Resources/models/yolov8/yolov8s.pt")

    frame = cv2.imread("e:/Resources/Municipio/drone/zona 2/odm_orthophoto/odm_orthophoto.tif")


    if (frame is None):
        exit()
        
    height = frame.shape[0]
    width = frame.shape[1]

    xOffset = 500
    #overlap = 100

    yres = []
    # detect all
    while xOffset < width - 600:

        yOffset = 500

        while yOffset < height - 600:

            roi = frame[yOffset:yOffset + 600, xOffset:xOffset + 600]
            
            renderF= roi.copy()

            if roi is None:
                break

            ys.detect(roi)

            if len(ys.objects)>0:
                ys.drawOwnDetections(renderF)

                for o in ys.objects:
                    if o.className == "broccoli" or o.className == "potted plant":
                        o.bbox[0] = o.bbox[0] + xOffset
                        o.bbox[1] = o.bbox[1] + yOffset
                        o.bbox[2] = o.bbox[2] + xOffset
                        o.bbox[3] = o.bbox[3] + yOffset 
                        yres.append(o)
    
                cv2.imshow("frame", renderF)

                renderAllDetections(frame, yres, 10)
                cv2.waitKey(1)

            yOffset += 500

            if len(yres) > 120:
                break

        xOffset += 500

        if len(yres) > 120:
                break


    m = renderAllDetections(frame, yres, 1)

    cv2.imwrite("../out_detected.jpg", m)

#open point shapefile
path = 'e:/Resources/Municipio/Drone/'
pointData = gpd.read_file(path + '/drone_explore_area.geojson')
print('CRS of Point Data: ' + str(pointData.crs))


# Opening JSON file
f = open(path + '/drone_explore_area.geojson')
geojson = json.load(f)

#open raster file
tandilRaster = rasterio.open(path + 'zona 2/odm_orthophoto/odm2.tif')
print('CRS of Raster Data: ' + str(tandilRaster.crs))
print('Number of Raster Bands: ' + str(tandilRaster.count))
print('Interpretation of Raster Bands: ' + str(tandilRaster.colorinterp))




clipper = ImageForClipModel(image_address=path + 'zona 2/odm_orthophoto/odm2.tif')
clipper.clip_raster(height=1000,
    width=1000,
    buffer=0,
    save_mode=False,
    prefix='raster_clip/clipped_band_',
    pass_empty=False)

index = 0
for clip in clipper.image_data:
    
    renderC = clip['crop']
    mp = (int(50),int(50))

    cv2.putText(renderC, f"{index}",mp,
                                cv2.FONT_HERSHEY_COMPLEX, 1.2, (250, 220, 255), 2, cv2.LINE_AA)
    cv2.imshow('clip', renderC)

    cv2.waitKey(20)
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