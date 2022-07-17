from turtle import width
import cv2 as cv
import numpy as np

#load mask
net = cv.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "mask_rcnn.pbtxt")

#generate random color for RGB from 0 to 255 of 80 objects
colors = np.random.randint(0, 255, (80, 3))

#load image
img = cv.imread('Cars.jpg')
img = cv.resize(img, (0,0), None, 0.2, 0.2)
height, width, _ = img.shape

#create black image
black_image = np.zeros((height, width, 3), np.uint8)
black_image[:] = (100, 50, 50)

# Detetc objects
blob = cv.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)

#reference mask_rcnn
boxes, masks = net.forward(["detection_out_final","detection_masks"])
count_detection = boxes.shape[2]
print(count_detection)

#iterate for all objects counted in the image
for i in range(count_detection):

    box = boxes[0, 0, i]
    obj_class_id = box[1]
    score = box[2]

    #set confidence threshold value for an object to appear
    if score < 0.5:
        continue

    #create the array for the rectangle box coordinate
    x = int(box[3] * width)
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)

    #region of interest extraction in (x, y) format
    #if swtched to black_image, the background changes
    roi = black_image[y: y2, x: x2]
    roi_height, roi_width, _ = roi.shape

    #Get object mask and print out mask size in pixel format
    mask = masks[i, int(obj_class_id)]

    #resize object mask & print mask value
    mask = cv.resize(mask, (roi_width, roi_height))
    print(mask)

    #create a threshold that define the range of mask
    #determined object or background base on binary image format
    _, mask = cv.threshold(mask, 0.5, 255, cv.THRESH_BINARY)

    #display rectangle box
    cv.rectangle(img, (x,y), (x2, y2), (255, 0, 0), 3)

    #get mask coordinate and count contour
    #extract boundaries of the white area in the mask
    contours, _ = cv.findContours(np.array(mask, np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    color = colors[int(obj_class_id)]
    for cnt in contours:
        print(cnt)
        cv.fillPoly(roi, [cnt], (int(color[0]),int(color[1]), int(color[2]))) #(0, 50, 0) format for specific color range example


    #extract every object by object within object mask(ie the rectangle box)
    #cv.imshow("ROI", roi)
    #cv.imshow("BIN_mask", mask)
    #cv.waitKey(0)
  
print(x, y)

cv.imshow('Cars,', img)
cv.imshow('Black_Cars', black_image)    #will output roi if black_image is called instead of img
cv.waitKey(0)