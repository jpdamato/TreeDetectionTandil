#https://pysource.com/2023/02/21/yolo-v8-segmentation
from ultralytics import YOLO
import numpy as np
import cv2
import threading
import time

##############################################
## class for describing detected object
class Object:
  def __init__(self,frame, bbox = None, classID = 0, className = "",segmentation = None, score = 0):
        self.bbox = bbox
        self.class_id  = classID
        self.className = className
        self.score = score
        self.segmentation = segmentation
        self.center = (int(self.bbox[0]*0.5+self.bbox[2]*0.5),int(self.bbox[1]*0.5+self.bbox[3]*0.5))
        if self.bbox is not None:          
          self.crop = frame[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]


######################################################
class YOLODetector:
    
   
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.bboxes = None 
        self.class_ids = None 
        self.segmentations = None
        self.scores = None
        self.objects = None
        self.names = self.model.model.names

    def postprocess(self, bboxes,  overlapThresh = 0.4):
        #return an empty list, if no boxes given
        if len(bboxes) == 0:
            return []
        x1 = boxes[:, 0]  # x coordinate of the top-left corner
        y1 = boxes[:, 1]  # y coordinate of the top-left corner
        x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
        y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
            # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We have a least a box of one pixel, therefore the +1
        indices = np.arange(len(x1))
        for i,box in enumerate(boxes):
            temp_indices = indices[indices!=i]
            xx1 = np.maximum(box[0], boxes[temp_indices,0])
            yy1 = np.maximum(box[1], boxes[temp_indices,1])
            xx2 = np.minimum(box[2], boxes[temp_indices,2])
            yy2 = np.minimum(box[3], boxes[temp_indices,3])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / areas[temp_indices]
            if np.any(overlap) > treshold:
                indices = indices[indices != i]
        return boxes[indices].astype(int)
    
    def detect(self, img):
        # Get img shape

        height, width, channels = img.shape

        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []
        if result.masks is not None:
            for seg in result.masks.segments:
                # contours
                seg[:, 0] *= width
                seg[:, 1] *= height
                segment = np.array(seg, dtype=np.int32)
                segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # Get class ids
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        # Get scores
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

        self.bboxes = bboxes
        self.class_ids = class_ids
        self.segmentations = segmentation_contours_idx
        self.scores = scores
        # convert in object format
        self.objects = []
        if len(segmentation_contours_idx) == 0 :
            for bbox, class_id,  score in zip(bboxes, class_ids,  scores):
                self.objects.append(Object(img,bbox, class_id,self.names[class_id], None, score))
        else:
            for bbox, class_id, seg, score in zip(bboxes, class_ids, segmentation_contours_idx, scores):
                self.objects.append(Object(img,bbox, class_id,self.names[class_id], seg, score))
        
        return self.objects
    
    def drawDetections(self, frame, objects):
        if objects is not None:
            for obj in objects:
            # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
                (x, y, x2, y2) = obj.bbox
                if obj.class_id >= 0:
                    cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
                    if obj.segmentation is not None:
                        cv2.polylines(frame, [obj.segmentation], True, (0, 0, 255), 4)

                    cv2.putText(frame, obj.className+ ':' + str(obj.score), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    def drawOwnDetections(self, frame):
        self.drawDetections(frame, self.objects )


######################################################
class YoloThread (threading.Thread):
    def __init__(self, threadID, model):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.model = model
        self.frame = None
    
    # return last inferences
    def infer(self, frame):
        self.frame = frame
        return self.model.bboxes ,        self.model.class_ids ,  self.model.segmentations,  self.model.scores 

    def run(self):
        while True:
            if (self.frame is not None):
            #    print( "Starting " + self.name)      
                self.model.detect(self.frame)
             #   print( "Exiting " + self.name)

            time.sleep(0.01)