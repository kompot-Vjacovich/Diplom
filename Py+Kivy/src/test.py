import numpy as np
import cv2

im = cv2.imread('dog.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(imgray,(5,5),0)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(im, contours, -1, (0,255,0), 3)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()