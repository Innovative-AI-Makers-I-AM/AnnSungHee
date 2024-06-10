import cv2
import os
import dlib
import numpy as np

# Dlib의 얼굴 탐지기와 랜드마크 예측기를 로드합니다.
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dlib의 얼굴 인식 모델 로드
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# 사진 저장을 위한 폴더 생성
if not os.path.exists("captured_photos"):
    os.makedirs("captured_photos")

# 이전에 촬영한 사진들의 얼굴 특징 벡터를 저장할 리스트
known_face_encodings = []
known_face_names = []

# 비디오 캡처를 시작합니다.
cap = cv2.VideoCapture(0)

# 사진 촬영 여부 플래그
photo_taken = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴을 탐지합니다.
    face_locations = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 탐지된 얼굴이 있고 사진이 아직 찍히지 않았다면 사진을 찍습니다.
    if len(face_locations) > 0 and not photo_taken:
        photo_filename = f"captured_photos/photo_{len(os.listdir('captured_photos'))}.jpg"
        cv2.imwrite(photo_filename, frame)
        print(f"Photo saved: {photo_filename}")

        # 현재 촬영한 사진의 얼굴 특징 벡터 추출
        face_img = cv2.imread(photo_filename)
        face_locations = dlib.get_frontal_face_detector(face_img)
        face_encoding = np.array([face_recognition_model.compute_face_descriptor(face_img, face_locations[0])])

        # 이전에 촬영한 사진들과 비교
        for known_encoding in known_face_encodings:
            distance = np.linalg.norm(face_encoding - known_encoding)
            similarity_score = 1 - (distance / 2)
            if similarity_score > 0.5:
                print(f"Match found! Similarity score: {similarity_score:.2f}")
                break
        else:
            # 새로운 얼굴이라면 리스트에 추가
            known_face_encodings.append(face_encoding)
            known_face_names.append(f"Person {len(known_face_names) + 1}")

        photo_taken = True

    # 얼굴 영역에 사각형을 그립니다.
    for (x, y, w, h) in face_locations:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 프레임을 출력합니다.
    cv2.imshow("Frame", frame)

    # 'q' 키를 누르면 루프를 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처와 모든 창을 종료합니다.
cap.release()
cv2.destroyAllWindows()
