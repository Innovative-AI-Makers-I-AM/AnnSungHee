import cv2
import numpy as np
import dlib

# 얼굴 탐지기 초기화
face_detector = dlib.get_frontal_face_detector()

# 얼굴 특징 벡터 추출 모델 초기화
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# 알려진 얼굴 특징 벡터 저장
known_face_encodings = []
known_face_names = []

# 실시간 영상 처리
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    
    # 얼굴 탐지
    face_locations = face_detector(frame)
    
    # 얼굴 특징 벡터 추출
    for face_location in face_locations:
        face_img = frame[face_location.top():face_location.bottom(), face_location.left():face_location.right()]
        face_encoding = np.array([face_recognition_model.compute_face_descriptor(face_img)])
        
        # 얼굴 인식
        distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        min_distance_index = np.argmin(distances)
        if distances[min_distance_index] < 0.6:
            name = known_face_names[min_distance_index]
        else:
            name = "Unknown"
        
        # 결과 출력
        cv2.rectangle(frame, (face_location.left(), face_location.top()), (face_location.right(), face_location.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, name, (face_location.left(), face_location.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
