import cv2
import os
import numpy as np
from face_recognition import face_encodings, compare_faces

# Dlib의 얼굴 탐지기와 랜드마크 예측기를 로드합니다.
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 사진 저장을 위한 폴더 생성
if not os.path.exists("captured_photos"):
    os.makedirs("captured_photos")

# 비디오 캡처를 시작합니다.
cap = cv2.VideoCapture(0)

# 사진 촬영 여부 플래그
photo_taken = False
photo_filename = None  # 여기에서 photo_filename 변수를 초기화합니다.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴을 탐지합니다.
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 탐지된 얼굴이 있고 사진이 아직 찍히지 않았다면 사진을 찍습니다.
    if len(faces) > 0 and not photo_taken:
        photo_filename = f"captured_photos/photo_{len(os.listdir('captured_photos'))}.jpg"
        cv2.imwrite(photo_filename, frame)
        print(f"Photo saved: {photo_filename}")
        photo_taken = True

    # 얼굴 영역에 사각형을 그립니다.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 프레임을 출력합니다.
    cv2.imshow("Frame", frame)

    # 'q' 키를 누르면 루프를 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if photo_filename:
    current_frame = cv2.imread(photo_filename)

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

# 비디오 캡처와 모든 창을 종료합니다.
cap.release()
cv2.destroyAllWindows()
