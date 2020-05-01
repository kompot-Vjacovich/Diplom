#import OpenCV package
import cv2
#import Recognition packages
import pytesseract as tesseract
import os
from PIL import Image as IM
#import other packages
import math
import numpy as np
import difflib
import re
from sklearn.cluster import KMeans


# Создание класса для хранения городов
class City():
    def __init__(self, image):
        self.img = image
        self.kp, self.desc = sift.detectAndCompute(image, None)
        
# Вычисление дескрипторов для заготовок городов
MIN_MATCH_COUNT = 15
sift = cv2.xfeatures2d.SIFT_create()
Simf = City(cv2.imread('test.jpg',0))
# Feature matching
index_params = dict(algorithm = 0, trees = 5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Расстояние между точками
def distance(p1,p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Определение схожести строк
def similarity(s1, s2):
    normalized1 = s1.lower()
    normalized2 = s2.lower()
    matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
    return matcher.ratio()

# Извлечение текста из изображения
def getTextWithTesseract(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,100,200)

    # Для распознавания текста как одного слова
    custom_oem_psm_config = r'--oem 3 --psm 6 bazaar'
    # Поиск слова и разделение его на символы с координатами каждого из них
    dirty = tesseract.image_to_boxes(edges, lang="rus", config=custom_oem_psm_config).split('\n')
    allSymb = list(map((lambda x: x.split(' ')), dirty))
    # Удаление не кириллических символов
    letters = list(filter(lambda x: len(x[0]) > 0 and ord(x[0]) in range(1040, 1103), allSymb))
    # Сортировка по х-координате левого нижнего угла буквы
    inRorder = sorted(letters, key=lambda sym: int(sym[1]))
    # Выборка из координат левого верхнего угла каждого символа
    dots = list(map(lambda x: [int(x[1]), int(x[2])], inRorder))
    # # Находим среднее растояние между двумя соседними буквами
    # dist = []
    # mean = 0
    # for i in range(len(dots)-1):
    #     dist.append(distance(dots[i], dots[i+1]))
    # if len(dist):
    #     mean = round(np.mean(dist))
    # dots = np.array(dots)
    # if len(dots):
    #     kmeans = KMeans(n_clusters=2, random_state=0).fit(dots)
    #     print(kmeans.labels_)
    
    text = ''.join(list(map(lambda x: x[0], inRorder)))

    return text

# Проекция 3D модели в кадр
def pasteModelIntoFrame(city, frame):
    kp1 = city.kp
    desc1 = city.desc
    final_image = frame
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, desc2 = sift.detectAndCompute(grayframe, None)
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

    if len(good)>MIN_MATCH_COUNT:
        query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        train_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        if matrix is not None:
            # Perspective transform
            h, w = city.img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            final_image = cv2.polylines(final_image, [np.int32(dst)], True, (255, 0, 0), 2)
    

    return final_image

def app():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()

        #tesseract
        text = getTextWithTesseract(frame)


        if similarity(text, "Симферополь") > 0.5:
            text = "Симферополь"
            frame = pasteModelIntoFrame(Simf, frame);
        else:
            text += " Плохо"

        print(text)
        print('------')
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    app()
    