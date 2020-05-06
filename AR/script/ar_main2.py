import argparse

import cv2
import numpy as np
import math
import os
from objloader_simple import *

# Минимальное количество совпадений, которые необходимо найти,
# чтобы считать распознование действительным
MIN_MATCH_COUNT = 15


def main():
    # Исходная модель
    img1 = cv2.imread('answer/test2.jpg')
    # 3D модель в формате OBJ
    obj = OBJ('models/test.obj', swapyz=True)
    # Матрица параметров камеры 
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    # Feature matching
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    cap = cv2.VideoCapture(0)
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
                    # Получение матрицы 3D-проекции из параметров матрицы гомографии и камеры
                    projection = projection_matrix(camera_parameters, matrix)  
                    # Проектирование модели
                    final_image = render(frame, obj, projection, img1, False)

            #Выведение результатов
            cv2.imshow('frame', final_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
