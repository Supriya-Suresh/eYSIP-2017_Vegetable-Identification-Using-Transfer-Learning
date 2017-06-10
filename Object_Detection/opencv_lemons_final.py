import cv2
import numpy as np
import matplotlib.pyplot as plt


# load the image, clone it for output, and then 
img = cv2.imread('C:\Users\sathish\Desktop\IIT Bombay Internship\lemon\scene00010.jpg')

#Blurring the image
img = cv2.blur(img,(5,5),-3)
#img = cv2.blur(img,(5,5),-5)
#img = cv2.medianBlur(img,5)

#Converting RGB Image into Grayscale Image
cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Using Hough transform technique 
circles = cv2.HoughCircles(cimg,cv2.cv.CV_HOUGH_GRADIENT,2,15,
                            param1=190,param2=50,minRadius=20,maxRadius=23)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    circle = cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

#Displaying the Image and Counting it's Quantity
cv2.imshow('detected circles',cimg)
Count =  "%d Number of Lemons."%(len(circles[0]))
print(Count)
cv2.waitKey(0)
cv2.destroyAllWindows()


