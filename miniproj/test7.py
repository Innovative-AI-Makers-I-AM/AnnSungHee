# 사진 비교
import cv2
import os
import numpy as np
from face_recognition import face_encodings, compare_faces

# 현재 찍힌 사진 로드
current_frame = cv2.imread("captured_photos/photo_0.jpg")

# 비교 대상 사진 로드
compare_image = cv2.imread("captured_photos/human.jpg")

# 얼굴 특징 추출
current_face_encoding = face_encodings(current_frame)[0]
compare_face_encoding = face_encodings(compare_image)[0]

# 얼굴 비교
results = compare_faces([current_face_encoding], compare_face_encoding)

# 결과 출력
if results[0]:
    print("Same person")
else:
    print("Different person")
