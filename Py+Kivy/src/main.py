# import kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
# import OpenCV package
import cv2
# import Recognition package
import pytesseract as tesseract
# import threading package
import continuous_threading as ct
# import other packages
import numpy as np
import difflib
import re
import copy

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
th1 = ''
th2 = ''

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

    text = ''.join(list(map(lambda x: x[0], inRorder)))

    return text

# Проекция 3D модели в кадр
def pasteModelIntoFrame(city, frame, goodImg):
    kp1 = city.kp
    desc1 = city.desc
    final_image = frame
    grayframe = cv2.cvtColor(goodImg, cv2.COLOR_BGR2GRAY)
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
    

    return final_image

class CamApp(App):

    def build(self):
        self.img1=Image()
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.img1)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        self.city = Simf
        _, self.goodImg = self.capture.read()
        Clock.schedule_interval(self.update, 1.0/60.0)
        th1 = ct.PeriodicThread(1.0/60.0, self.update)
        th1.start()
        th2 = ct.PeriodicThread(1.0, self.recognition)
        th2.start()
        return layout

    def update(self, dt):
        _, frame = self.capture.read()

        frame = pasteModelIntoFrame(self.city, frame, self.goodImg);
       
        buf1 = cv2.flip(frame, 0)        

        width = self.img1.size[0]
        height = int(frame.shape[0]*width/frame.shape[1])
        dim = (width, height)
        resized = cv2.resize(buf1, dim, interpolation = cv2.INTER_AREA)

        buf = resized.tostring()
        texture1 = Texture.create(size=(resized.shape[1], resized.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.img1.texture = texture1

    def recognition(self):
        _, self.goodImg = self.capture.read()

        text = getTextWithTesseract(self.goodImg).lower()
        print(text)
        if similarity(text, "cимферополь") > 0.5 or text.find("симферополь") != -1:
            text = "Симферополь"
            self.city = Simf
        elif similarity(text, "севастополь") > 0.5 or text.find("севастополь") != -1:
        	text = "Севастополь"
            self.city = Sevas
        elif similarity(text, "керчь") > 0.5 or text.find("керчь") != -1:
        	text = "Керчь"
            self.city = Kerch
        elif similarity(text, "судак") > 0.5 or text.find("судак") != -1:
        	text = "Судак"
            self.city = Sudak
        elif similarity(text, "ялта") > 0.5 or text.find("ялта") != -1:
        	text = "Ялта"
            self.city = Yalta
        else:
            text += " Плохо"

        print(text)
if __name__ == '__main__':
    CamApp().run()