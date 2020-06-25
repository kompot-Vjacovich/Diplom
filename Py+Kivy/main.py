# import kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
# import Recognition package
import pytesseract as tesseract
# import threading package
import continuous_threading as ct
# import other packages
from fuzzywuzzy import fuzz
# my packages
from src.objloader import *
from src.render import *

# Создание класса для хранения городов
class City():
    def __init__(self, image, model):
        self.img = cv2.imread(image,0)
        self.kp, self.desc = sift.detectAndCompute(self.img, None)
        self.model = OBJ(model, swapyz=True)

Simf = City('ref/Simf.jpg', 'models/test.obj')
Sevas = City('ref/Sevas.jpg', 'models/Sevas.obj')
Yalta = City('ref/Yalta.jpg', 'models/Yalta.obj')
Kerch = City('ref/Kerch.jpg', 'models/Kerch.obj')
Sudak = City('ref/Sudak1.jpg', 'models/Sudak.obj')
th = ''

# Определение схожести строк
def similarity(s1, s2):
    if len(s1) < len(s2)/2:
        return 0

    dictionary = {
        'симферополь': 'CUM@EPONO1b',
        'керчь': 'KEPUb',
        'севастополь': 'CEBACTONO1b',
        'судак': 'CYMAK',
        'ялта': 'SITA'
    }
    match = fuzz.partial_ratio(dictionary[s2], s1.upper())
    return match/100


# Извлечение текста из изображения
def getTextWithTesseract(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,100,200)
    
    # Для распознавания текста как одного слова
    custom_oem_psm_config = r'--oem 3 --psm 7 bazaar'
    # Поиск слова и разделение его на символы с координатами каждого из них
    dirty = tesseract.image_to_string(edges)
    text = dirty.replace(' ', '')

    return text

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
        th = ct.PeriodicThread(1.0, self.recognition)
        th.start()
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

        text = getTextWithTesseract(self.goodImg)
        text = text.lower()
        print(text)
        if similarity(text, "симферополь") > 0.7:
            text = "Симферополь"
            self.city = Simf
        elif similarity(text, "севастополь") > 0.7:
            text = "Севастополь"
            self.city = Sevas
        elif similarity(text, "керчь") > 0.7:
            text = "Керчь"
            self.city = Kerch
        elif similarity(text, "судак") > 0.7:
            text = "Судак"
            self.city = Sudak
        elif similarity(text, "ялта") > 0.7:
            text = "Ялта"
            self.city = Yalta
        else:
            text += " Плохо"

        print(text)
if __name__ == '__main__':
    CamApp().run()