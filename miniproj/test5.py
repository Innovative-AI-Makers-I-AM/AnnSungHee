import cv2
import os
from skimage.metrics import structural_similarity as ssim

# Dlib의 얼굴 탐지기와 랜드마크 예측기를 로드합니다.
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 사진 저장 및 비교를 위한 폴더 생성
if not os.path.exists("captured_photos"):
    os.makedirs("captured_photos")

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
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 탐지된 얼굴이 있고 사진이 아직 찍히지 않았다면 사진을 찍습니다.
    if len(faces) > 0 and not photo_taken:
        photo_filename = f"captured_photos/photo_{len(os.listdir('captured_photos'))}.jpg"
        cv2.imwrite(photo_filename, frame)
        print(f"Photo saved: {photo_filename}")

        # 기존 사진과 비교
        if len(os.listdir('captured_photos')) > 1:
            prev_photo = cv2.imread(f"captured_photos/photo_{len(os.listdir('captured_photos'))-1}.jpg")
            similarity = ssim(gray, cv2.cvtColor(prev_photo, cv2.COLOR_BGR2GRAY))
            print(f"Similarity score: {similarity:.2f}")
            if similarity > 0.9:
                print("The photos are similar.")
            else:
                print("The photos are different.")

        photo_taken = True

    # 얼굴 영역에 사각형을 그립니다.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 프레임을 출력합니다.
    cv2.imshow("Frame", frame)

    # 'q' 키를 누르면 루프를 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처와 모든 창을 종료합니다.
cap.release()
cv2.destroyAllWindows()
