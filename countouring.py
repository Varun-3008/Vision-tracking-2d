import cv2 as cv 
import numpy as np 
from PIL import Image
import imutils
from get_hsv_values import get_limits # Set masking parameters through here

arrowlength = 50 # constant to describe how large the  arrow is in pixels.
cap = cv.VideoCapture(0) # open video capture device for camera
colour = [255 ,0,0] #constant to select what colour to mask *(IN BGR)*

# sets the frame dimensions *IDEALLY SET CAMERA RES TO AVOID DIGITAL STRETCH BEFORE DISPLAY* 
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# processes images to have basic required axis and mask.
while True:
    isTrue, frame = cap.read()
    
    hsvimg = cv.cvtColor(frame, cv.COLOR_BGR2HSV) # converts RGB TO HSV images
    lowerLimit, upperLimit =  get_limits(colour) #calls function to set limits
    mask = cv.inRange(hsvimg, lowerLimit, upperLimit) # masking function
    
    # generates contour based on masked image and method specified.
    contour =cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  
    contour = imutils.grab_contours(contour)
    
    # setup of axis and central rectangle
    cv.arrowedLine(frame,(320,240),(320-arrowlength,240),[255,0,0],2)
    cv.arrowedLine(frame,(320,240),(320,240-arrowlength),[0,0,255],2)
    cv.putText(frame,"x",(320-arrowlength-10,240-10),cv.FONT_HERSHEY_SIMPLEX,0.5,[255,0,0])
    cv.putText(frame,"y",(320+10,240-arrowlength),cv.FONT_HERSHEY_SIMPLEX,0.5,[0,0,255])
    cv.rectangle(frame,(320-20,240+20),(320+20,240-20),[0,255,255],1)
    
    # for loop works throuh points in contours which fit mask to create area.
    for c in contour:
        area = cv.contourArea(c)
        
        cv.drawContours(frame,[c],-1,(0,255,0),3)
        if area > 5000:
            moment = cv.moments(c)
            # calculates centroid based of moments on x and y axis. 
            cx = int(moment["m10"]/moment["m00"])
            cy = int(moment["m01"]/moment["m00"])

            #generates centroid
            cv.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            cv.putText(frame,"centroid", (cx-20,cy-20),cv.FONT_HERSHEY_SIMPLEX,0.5,[255,255,255])
            if cx < 300:
                print("move left")
                # rightmotorspeed > leftmotorspeed
            elif cx >340:
                print("move right")
                # leftmotorspeed > rightmotorspeed 
            elif 300<cx<340:
                print("stop")

    #displays images 
    cv.imshow('masked', cv.flip(cv.resize(frame,(960,720)),1))

    if cv.waitKey(10) & 0xFF==ord(' '):
        break
    
cap.release()
cv.destroyAllWindows()