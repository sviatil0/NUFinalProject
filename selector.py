from picamera.array import PiRGBArray     #As there is a resolution problem in raspberry pi, will not be able to capture frames by VideoCapture
from picamera import PiCamera
import RPi.GPIO as GPIO
import time
import cv2
import cv2 as cv
import numpy as np

#hardware work
GPIO.setmode(GPIO.BCM) 

#Image analysis work
def segment_colour_wooden(frame):   
    hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    lower_yellow = np.array([11,85,199])
    upper_yellow = np.array([179,255,255])

    mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

    # Doing to clear the white noises
    kern_dilate = np.ones((8,8),np.uint8)
    kern_erode  = np.ones((3,3),np.uint8)
    
    mask= cv2.erode(mask,kern_erode)      #Eroding
    mask=cv2.dilate(mask,kern_dilate)     #Dilating
    return mask

def segment_colour_white(frame):    
    hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    lower_yellow = np.array([11,85,199])
    upper_yellow = np.array([179,255,255])

    mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

    # Doing to clear the white noises
    kern_dilate = np.ones((8,8),np.uint8)
    kern_erode  = np.ones((3,3),np.uint8)
    
    mask= cv2.erode(mask,kern_erode)      #Eroding
    mask=cv2.dilate(mask,kern_dilate)     #Dilating
    return mask

def find_blob(blob): #returns the red colored circle
    largest_contour=0
    cont_index=0
    contours, hierarchy = cv2.findContours(blob, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if (area >largest_contour) :
            largest_contour=area
           
            cont_index=idx
                              
    r=(0,0,2,2)
    if len(contours) > 0:
        r = cv2.boundingRect(contours[cont_index])
       
    return r,largest_contour


def target_hist(frame):
    hsv_img=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
    hist=cv2.calcHist([hsv_img],[0],None,[50],[0,255])
    return hist

#CAMERA CAPTURE
#initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (160, 120)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=(160, 120))

# allow the camera to warmup
time.sleep(0.001)
 

for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
      #grab the raw NumPy array representing the image, then initialize the timestamp and occupied/unoccupied text
      frame = image.array
      frame=cv2.flip(frame,1)
      # cv2.imshow('initiall image',frame)
      global centre_x
      global centre_y
      centre_x=0.
      centre_y=0.
      hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Work with different colors starts
    

    # Trying white color
      mask_white=segment_colour_white(frame)      
      loct,area=find_blob(mask_white)
      x,y,w,h=loct

      if (w*h) < 10:
        

        # Trying wooden color
        mask_wooden=segment_colour_wooden(frame)      
        loct,area=find_blob(mask_wooden)
        x,y,w,h=loct
        if(w*h)<10:
            print("It's silver")
        else:
            print("It's wooden")

      else:
        print("It's white")