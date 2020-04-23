import numpy as np
import cv2 as cv
MIN_MATCH_COUNT = 15
img1 = cv.imread('test.jpg',0)          # queryImage
cap = cv.VideoCapture(0)
sift = cv.xfeatures2d.SIFT_create()
kp1, desc1 = sift.detectAndCompute(img1, None)
# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv.FlannBasedMatcher(index_params, search_params)

while True:
    _, img2 = cap.read()
    grayframe = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)  # trainimage
    kp2, desc2 = sift.detectAndCompute(grayframe, None)
    matches = flann.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        train_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        if matrix is not None:
            # Perspective transform
            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, matrix)
            img2 = cv.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None)
    cv.imshow('gray', img3)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()