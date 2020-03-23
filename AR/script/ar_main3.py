
import argparse

import cv2
import numpy as np
import math
import os
from objloader_simple import *

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 10  


def main():
    cap = cv2.VideoCapture(0)
    while True:
        frame = cap.read()     
        model = cv2.imread('answer/fuck.jpg', 0)
        # ORB keypoint detector
        orb = cv2.ORB_create()              
        # create brute force  matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
        # Compute model keypoints and its descriptors
        kp_model, des_model = orb.detectAndCompute(model, None)  
        # Compute scene keypoints and its descriptors
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        # Match frame descriptors with model descriptors
        matches = bf.match(des_model, des_frame)
        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > MIN_MATCHES:
            # draw first 15 matches.
            frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:MIN_MATCHES], 0, flags=2)
            # show result
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
        else:
            print("Not enough matches have been found - %d/%d" % (len(matches), MIN_MATCHES))

if __name__ == '__main__':
    main()
