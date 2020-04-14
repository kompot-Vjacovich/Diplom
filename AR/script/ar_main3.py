import argparse

import cv2
import numpy as np
import math
import os
from objloader_simple import *

# Минимальное количество совпадений, которые необходимо найти,
# чтобы считать распознование действительным
MIN_MATCHES = 45  


def main():
    homography = None
    # Текущая директория, учитывая особенности ОС
    dir_name = os.getcwd()
    # Исходная модель
    model = cv2.imread('answer/fuck.jpg')
    # 3D модель в формате OBJ
    obj = OBJ('models/pirate.obj', swapyz=True)
    # Матрица параметров камеры 
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # Детектор ключевых точек ORB
    orb = cv2.ORB_create()              
    # Создание объекта для "грубого" перебора используя расстояние Хемминга
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
    # Вычисление ключевых точек модели и её дескрипторов
    kp_model, des_model = orb.detectAndCompute(model, None) 
    # Записываем видео
    cap = cv2.VideoCapture(0)
    while True:  
        # Считываем текущий кадр
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return   
        # Вычисление ключевых точек сцены и её дескрипторов
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        # Сопоставление дескрипторов сцены и дескрипторов модели
        matches = bf.match(des_model, des_frame)
        # Сортировка их в порядке удалённости
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > MIN_MATCHES:
            # Определение различий между исходными точками и точками сцены
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            # Вычисление гомографии
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if args.matches:
                # Рисование первых MIN_MATCHES совпадений
                frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:MIN_MATCHES], 0, flags=2)
            if args.rectangle:
                # Рисование прямоугольника(рамки) вокруг найденной модели
                h, w, channels = model.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # Расставление угловых точек
                dst = cv2.perspectiveTransform(pts, homography)
                # Соединение их линиями  
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            if args.model and homography is not None:
                try:
                    # Получение матрицы 3D-проекции из параметров матрицы гомографии и камеры
                    projection = projection_matrix(camera_parameters, homography)  
                    # Проектирование модели
                    # print("start")
                    # print(obj)
                    frame = render(frame, obj, projection, model, False)
                    #frame = render(frame, model, projection)
                except:
                    pass
            if (frame.shape[1] > 1200) or (frame.shape[0] > 700):
                # Процент от изначального размера
                scale_percent = 70 
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                dim = (width, height)
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            #Выведение результатов
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Not enough matches have been found - %d/%d" % (len(matches), MIN_MATCHES))

    cap.release()
    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, model, color=False):
    """
    Рендер 3D модели на кадр
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3)
    h, w, channels = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # Визуализация модели в середине опорной поверхности. 
        # Для этого точки модели должны быть смещены
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    Вычисление матрицы 3D-проекции из калибровочной матрицы камеры и расчетной гомографии.
    """
    # Вычисление поворота и смещения по осям x и y
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # Нормирование векторов
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # Вычисление ортогонального базиса
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # Вычисление 3D-проекции модели на текущий кадр
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Конвертирование hex строки в RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
parser.add_argument('-mo','--model', help = 'Specify model to be projected', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()
