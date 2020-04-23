#import kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
#import OpenCV package
import cv2
#import Recognition packages
import pytesseract as tesseract
import os
from PIL import Image as IM
#import other packages
import numpy
import difflib

def similarity(s1, s2):
  normalized1 = s1.lower()
  normalized2 = s2.lower()
  matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
  return matcher.ratio()

def getTextWithTesseract(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,100,200)
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, edges)
    text = tesseract.image_to_string(IM.open(filename), lang="rus")
    os.remove(filename)

    if similarity(text, "Симферополь") > 0.5:
        text = "Симферополь"
    else:
        text = "Плохо виден текст"

    return text

class CamApp(App):

    def build(self):
        self.img1=Image()
        layout = BoxLayout(orientation="vertical")
        self.label = Label()
        layout.add_widget(self.label)
        layout.add_widget(self.img1)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        # cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def update(self, dt):
        # display image from cam in opencv window
        ret, frame = self.capture.read()
        # cv2.imshow("CV2 Image", frame)

        # convert it to texture
        buf1 = cv2.flip(frame, 0)

        #tesseract
        text = getTextWithTesseract(frame)
        self.label.text = text
        self.label.size_hint = (1, 0.2)

        # display image from the texture
        width = self.img1.size[0]
        height = int(frame.shape[0]*width/frame.shape[1])
        dim = (width, height)
        resized = cv2.resize(buf1, dim, interpolation = cv2.INTER_AREA)

        buf = resized.tostring()
        texture1 = Texture.create(size=(resized.shape[1], resized.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.img1.texture = texture1
        self.img1.allow_stretch = True
        self.img1.keep_ratio = False
        #cv2.imshow("test", edges)

if __name__ == '__main__':
    CamApp().run()
    # cv2.destroyAllWindows()