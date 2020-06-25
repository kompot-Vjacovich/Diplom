import numpy as np
import cv2
import pytesseract as tesseract
from fuzzywuzzy import fuzz

cap = cv2.VideoCapture(0)

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
    match = fuzz.partial_ratio(dictionary[s2], s1)
    return match/100

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,100,200)
    
    # Для распознавания текста как одного слова
    custom_oem_psm_config = r'--oem 3 --psm 7 bazaar'
    # Поиск слова и разделение его на символы с координатами каждого из них
    dirty = tesseract.image_to_string(edges, lang="rus")
    text = dirty.replace(' ', '')
    # print(dirty)
    # dirty = dirty.split('\n')
    # allSymb = list(map((lambda x: list(x)), dirty))
    # # Удаление не кириллических символов
    # letters = list(map(lambda e: list(filter(lambda x: len(x) > 0 and ord(x) in range(1040, 1103), e)), allSymb))

    # text = ''.join(list(map(lambda x: ''.join(x), letters)))
    if similarity(text, "симферополь") > 0.5:
        text = "Симферополь"
    elif similarity(text, "севастополь") > 0.5:
        text = "Севастополь"
    elif similarity(text, "керчь") > 0.5:
        text = "Керчь"
    elif similarity(text, "судак") > 0.5:
        text = "Судак"
    elif similarity(text, "ялта") > 0.5:
        text = "Ялта"
    print(text)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()