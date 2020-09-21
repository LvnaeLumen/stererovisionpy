#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import pathlib

def nothing(*argv):
        pass
#finding distance to point
def measure_distance(x,y):
    average=0
    for u in range (-1,2):
        for v in range (-1,2):
            average += disparitySGBM[y+u,x+v] #using SGBM in area
    average=average/9
    distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
    #cubic equation from source (experimental)
    distance= np.around(distance*0.01,decimals=2)
    return distance


def coords_mouse_disp(event,x,y,flags,param): #Function measuring distance to object
    if event == cv2.EVENT_LBUTTONDBLCLK: #double leftclick on disparity map (control windwo)
        #print (x,y,disparitySGBM[y,x],sgbm_filteredImg[y,x])
        global distance_to_object_string
        distance_to_object_string = str(measure_distance(x,y))+" m"

#resizing image
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Resizing high definition images for faster evaluation
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the 0idth and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


num_disp = 128  ## global sgbm (and changheable trackbar) parameters
block_size = 3
window_size = 3
min_disp = 2
disp12 = 10
uniqR = 10
speckWS = 100
speckR = 32
confidence_threshold = 0.5
color_map = 0

#displaying on the depth map
distance_to_object = 0
distance_to_object_string = "? m"

#flags for pressed keys
flagObj = 0
flagP = 0
flagD = 0
flagB = 0
flagN = 0
flagCen = 0
#block updating object finding
refreshBL = 0
refreshBR = 0

obj_rects = []
obj_centers = []

def detectImage(image): #using Deep Neural Netrowk trained dataset to find objects
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers) #found something like real objects
    rects = []
    class_ids = []
    confidences = []
    center = []
    font = cv2.FONT_HERSHEY_PLAIN

    global confidence_threshold
    for out in outs: #in all found semi-objects
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id] #how much is it close to certain object
            if confidence > confidence_threshold: #if so, draw things
                # Object detected
                #dot coords
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                rects.append([x, y, w, h])
                center.append([center_x,center_y])
                confidences.append(float(confidence)) #add to founded objects
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(rects, confidences, confidence_threshold, 0.4)#get object

    for i in range(len(rects)): #name object and draw stuff on picture
        if i in indexes:
            x, y, w, h = rects[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 3, color, 3) #name of object
#    cv2.imshow(name,image)
    return image, rects, center

def updateTrackbar(): #function for reading and adjustring trackbar parameters
    global block_size
    global num_disp
    global min_disp

    global color_map
    global window_size
    global confidence_threshold

    min_disp = cv2.getTrackbarPos('minDisparity', 'control')
    newBSize = cv2.getTrackbarPos('blockSize', 'control')
    newNumDisp = cv2.getTrackbarPos('numDisparities', 'control')
    color_map = cv2.getTrackbarPos('Color Map', 'control')
    window_size = cv2.getTrackbarPos('windowSize', 'control')
    confidence_threshold = cv2.getTrackbarPos('confidence', 'control') / 10

    if (newBSize < 5):
        newBSize = 5;
    block_size = int( 2 * round( newBSize / 2. ))+1 #fool protection
    if newNumDisp < 16:
        newNumDisp=newNumDisp+16
    num_disp = int( 16 * round( newNumDisp / 16. )) #fool protection

def updateDisp():
    global stereoSGBM
    stereoSGBM = cv2.StereoSGBM_create(
        minDisparity = min_disp, # dynamic
        numDisparities=num_disp, # dynamic
        blockSize=block_size, # dynamic
        P1 = 8*3*window_size**2, # indirectly dynamic
        P2 = 32*3*window_size**2, # indirectly dynamic
        disp12MaxDiff = 1, ##no difference
        preFilterCap = 0, ##no difference
        uniquenessRatio = uniqR, #nd
        speckleWindowSize = speckWS, #nd
        speckleRange = speckR,#nd
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def updateFilter():
    lmbda = 80000
    sigma = 1.8
    visual_multiplier = 1.0
    #http://timosam.com/python_opencv_depthimage/
    #filtering image
    #https://docs.opencv.org/3.4/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html
    global sgbm_matcher
    global sgbm_wls_filter


    sgbm_matcher = cv2.ximgproc.createRightMatcher(stereoSGBM)#!
    sgbm_wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoSGBM)#!
    sgbm_wls_filter.setLambda(lmbda)
    sgbm_wls_filter.setSigmaColor(sigma)

def redrawDisp():
    global disparitySGBM

    disparitySGBM = stereoSGBM.compute(gray_left, gray_right)#.astype(np.float32) / 16.0

    sgbm_displ = disparitySGBM  # .astype(np.float32)/16 ##!
    sgbm_dispr = sgbm_matcher.compute(gray_right, gray_left)  # .astype(np.float32)/16
    sgbm_displ = np.int16(sgbm_displ)
    sgbm_dispr = np.int16(sgbm_dispr)

    sgbm_filteredImg = sgbm_wls_filter.filter(sgbm_displ, gray_left, None, sgbm_dispr)  # important to put "gray_left" here!!!
    sgbm_filteredImg = cv2.normalize(src=sgbm_filteredImg, dst=sgbm_filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    sgbm_filteredImg = np.uint8(sgbm_filteredImg)
    disparitySGBM= ((disparitySGBM.astype(np.float32)/ 16)-min_disp)/num_disp

    closing= cv2.morphologyEx(disparitySGBM,cv2.MORPH_CLOSE, kernel)

    dispc= (closing-closing.min())*255
    dispC= dispc.astype(np.uint8) #additional


    if(flagP == 1): # press spacebar to show colored matplotlib disparity, 'q' to close
        dispC= cv2.applyColorMap(dispC,color_map)
        # Change the Color of the Picture into an Ocean Color_Map
        sgbm_filteredImg= cv2.applyColorMap(sgbm_filteredImg,color_map)
    if(flagObj == 1): #precc d to show all rectangles of found objects and distance to their centers
        if(flagP == 0): #cant work with color mapped images
            sgbm_filteredImg = cv2.cvtColor(sgbm_filteredImg, cv2.COLOR_GRAY2RGB) #make rgb for colored rects
        for i in range(len(obj_rects)):
            x, y, w, h = obj_rects[i]
                #label = str(classes[class_ids[i]])
            color = colors[i]
            dist = measure_distance(obj_centers[i][0],obj_centers[i][1])
            cv2.putText(sgbm_filteredImg, str(dist)+ " m",
            (obj_centers[i][0] - 20, obj_centers[i][1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 3)
            cv2.rectangle(sgbm_filteredImg, (x, y), (x + w, y + h), color, 2)
            cv2.circle(sgbm_filteredImg, (obj_centers[i][0], obj_centers[i][1]), 20, color,  2)

    if(flagCen == 1): #press k to show distance to last doubleclicked dot
        cv2.putText(sgbm_filteredImg, distance_to_object_string,
            (sgbm_filteredImg.shape[1] - 200, sgbm_filteredImg.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 3)

    if(flagD == 1): #press v to show unfiltered disparity
        cv2.imshow('disp', dispC)
    else:
        cv2.destroyWindow('disp')

    cv2.imshow('control',sgbm_filteredImg)
    cv2.setMouseCallback("control",coords_mouse_disp,sgbm_filteredImg)

######################### MAIN PROGRAM

### ERROR CHECKING
try:
    sys.argv[1]
    sys.argv[2]
except IndexError:
     sys.exit('No file paths')

left_picture = pathlib.Path(sys.argv[1]) #reading image names from console parameters
right_picture = pathlib.Path(sys.argv[2])

if (not left_picture.is_file() or not right_picture.is_file()): #no files
    sys.exit('Wrong file path')

colored_left = cv2.imread(sys.argv[1]) #reading files
colored_right = cv2.imread(sys.argv[2])


height_left = colored_left.shape[0] #checking if files are of same size
height_right = colored_right.shape[0]
width_left = colored_left.shape[1]
width_right = colored_right.shape[1]

if( (height_left != height_right) and (width_left != width_right) ):
    sys.exit('Files are of different size')

### RESIZING IMAGES
if (width_left > 2*height_left ):
    colored_left = maintain_aspect_ratio_resize(colored_left, width = 1200)
    colored_right = maintain_aspect_ratio_resize(colored_right, width = 1200)
else:
    colored_left = maintain_aspect_ratio_resize(colored_left, height = 600)
    colored_right = maintain_aspect_ratio_resize(colored_right, height = 600)

height, width, channels = colored_left.shape ##for object detection

gray_left = cv2.cvtColor(colored_left, cv2.COLOR_BGR2GRAY) #have to work with gray images
gray_right = cv2.cvtColor(colored_right, cv2.COLOR_BGR2GRAY)


#### SETTING FOR CONTROL WINDOW
cv2.namedWindow('control')
cv2.createTrackbar('minDisparity', 'control', 2, 128, nothing)
cv2.createTrackbar('numDisparities', 'control', 128, 800, nothing)
cv2.createTrackbar('blockSize', 'control', 3, 135, nothing)
cv2.createTrackbar('windowSize', 'control', 3, 20, nothing)
cv2.createTrackbar('confidence', 'control',  4, 10, nothing)
cv2.createTrackbar('Color Map', 'control',  0, 11, nothing)


## SETTINGS FOR OBJECT DETECTION
# for object detection ###todo gpu
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
kernel= np.ones((3,3),np.uint8)


### SETTINGS FOR STEREO VISION AND FILTERING
stereoSGBM = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1, ##no difference
    preFilterCap = 0, ##no difference
    uniquenessRatio = uniqR,
    speckleWindowSize = speckWS,
    speckleRange = speckR,
    #mode = modeT)
    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)

lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

#filtering image (REVIEWED in special functions )
disparitySGBM = stereoSGBM.compute(gray_left, gray_right)#.astype(np.float32) / 16.0
sgbm_matcher = cv2.ximgproc.createRightMatcher(stereoSGBM)#!
sgbm_wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoSGBM)#!

sgbm_displ = disparitySGBM  # .astype(np.float32)/16 ##!
sgbm_dispr = sgbm_matcher.compute(gray_right, gray_left)
sgbm_displ = np.int16(sgbm_displ)
sgbm_dispr = np.int16(sgbm_dispr)
sgbm_filteredImg = sgbm_wls_filter.filter(sgbm_displ, gray_left, None, sgbm_dispr)

#getting first shot object detection
im1 = colored_left.copy()
im1, obj_rects, obj_centers = detectImage(im1)
while True:
    updateTrackbar()
    updateDisp()
    updateFilter()
    redrawDisp()
    if(flagB):
        if(refreshBL):
            im1 = colored_left.copy() #detectImage() is a destuctive function
            im1, obj_rects, obj_centers = detectImage(im1)
            refreshBL = 0
            cv2.imshow('Left Image', im1)
    else:
        cv2.destroyWindow('Left Image')

    if(flagN):
        if(refreshBR):
            im2 = colored_right.copy()
            im2,a,b= detectImage(im2)
            refreshBR = 0
            cv2.imshow('Right Image', im2)
    else:
        cv2.destroyWindow('Right Image')


    #print (flagB, flagN)
    #print (refreshBL, refreshBR)
    ch = cv2.waitKey(1)
    if ch == 32:
        flagP = not flagP
    if ch == ord('v'):
        flagD = not flagD
    if ch == ord('b'):
        flagB = not flagB
        refreshBL = 1
    if ch == ord('n'):
        flagN = not flagN
        refreshBR = 1
    if ch == ord('d'):
        flagObj = not flagObj
    if ch == ord('k'):
         flagCen = not flagCen
    if ch == 27:
        break
cv2.destroyAllWindows()
