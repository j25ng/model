from mtcnn import MTCNN
import cv2

# MTCNN 모델 불러오기
detector = MTCNN()

# 이미지 불러오기
image = cv2.imread('images/people.jpg')

# 얼굴 감지
faces = detector.detect_faces(image)

# 얼굴 부분 자르기 및 저장
for i, face in enumerate(faces):
    x, y, width, height = face['box']
    face_crop = image[y:y+height, x:x+width]
    cv2.imwrite(f'face_{i}.jpg', face_crop)

