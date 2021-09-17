#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 21:31:55 2021

@author: Mithra
"""

import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import pytesseract 


#Read the image 


img_path = ('/Users/Mithra/Desktop/Hand Hygiene Monitoring/ID.jpeg')
#img_path_2 = ('drive/My Drive/Reg_no.PNG')

img = cv2.imread(img_path)
img_cvt=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_cvt)
plt.show()

grey_scale_img = cv2.cvtColor(img_cvt, cv2.COLOR_BGR2GRAY)
plt.imshow(grey_scale_img, cmap = 'gray', interpolation = 'bicubic')
plt.title('Grey Scale Image')
plt.show( )

#filter out noise 
filter = cv2.bilateralFilter(grey_scale_img, 15, 75, 75)
# format (grey_scale_img, d, sigma colour, sigma colour)
# d = diameter of each pixel neighborhood  
# sigma colour is the value of sigma in the colour space, more the sigma, colours farther away from each other will get mixed
plt.imshow(filter)
plt.show()

canny_edges = cv2.Canny(grey_scale_img, 100, 170, apertureSize = 3)#display canny edges
plt.imshow(canny_edges)

edged_dup = canny_edges.copy() # duplicate image 
contours, hierarchy = cv2.findContours(edged_dup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#draw contours 
#hierarchy type is retr_external 
#stores all boundary points chain_appox_none
# for a square image where u know there arent a lot of contours use chain_approx_simple
x= cv2.drawContours(img_cvt, contours, -1, (255,0,0),4)
plt.imshow(x)
plt.show()

# Python program to illustrate 
# template matching 
import cv2 
import numpy as np 


# Show the final image with the matched area. 
cv2.imshow('Detected',img)

#crop image 
x1 = 160
y1 = 829
x2 = 490
y2 = 890
roi = img_cvt[y1:y2, x1:x2]
plt.imshow(roi)
plt.show()

#pytesseract.pytesseract.tesseract_cmd = r'/Users/Mithra/Library/Caches/pip/wheels/67/71/6c/7a8c5ca2e699752506999ae7baeb692e2b4fc6488c2cddcb22'
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'
#OCR
#extractedInformation = tesseract.image_to_string(roi)
extractedInformation = pytesseract.image_to_string(roi)
print('Registration number = '+ extractedInformation)
