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

class MainApp(App):
    def build(self):
        img = Image(source='C://Users/HP15/Pictures/IMG_1191.PNG',
                    pos_hint={'center_x':.5, 'center_y':.5})
 
        return img
 
if __name__ == '__main__':
    app = MainApp()
    app.run()