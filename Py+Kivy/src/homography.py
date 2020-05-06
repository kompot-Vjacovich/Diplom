import numpy as np
import cv2
MIN_MATCH_COUNT = 15
img1 = cv2.imread('test.jpg',0)          # queryImage
cap = cv2.VideoCapture(0)
sift = cv2.xfeatures2d.SIFT_create()
kp1, desc1 = sift.detectAndCompute(img1, None)
# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    _, frame = cap.read()
    final_image = frame
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, desc2 = sift.detectAndCompute(grayframe, None)
    if desc2 is not None:
        # Находим по 2 ближайших дескриптора для каждой точки
        # Два раза: маркер к картинке и обратно (они будут разными)
        matches1to2 = flann.knnMatch(desc1, desc2, k=2)
        matches2to1 = flann.knnMatch(desc2, desc1, k=2)
        # Выкидываем точки с менее чем двумя соответствиями
        matches1to2 = [x for x in matches1to2 if len(x) == 2]
        matches2to1 = [x for x in matches2to1 if len(x) == 2]
        # Выкидываем точки, в которых не сильно уверены
        ratio = 0.6
        good1to2 = [m for m,n in matches1to2 if m.distance < ratio * n.distance]
        good2to1 = list([m for m,n in matches2to1 if m.distance < ratio * n.distance])
        # Выкидываем несимметричные соответствия
        good = []
        for m in good1to2:
            for n in good2to1:
                if m.queryIdx == n.trainIdx and n.queryIdx == m.trainIdx:
                    good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            train_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            if matrix is not None:
                # Perspective transform
                h, w = city.img.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)
                final_image = cv2.polylines(final_image, [np.int32(dst)], True, (255, 0, 0), 3)
    
    cv2.imshow('gray', final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()