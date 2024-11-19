import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# EAR 계산 함수
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 얼굴 탐지기와 얼굴 랜드마크 예측기 초기화
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')  # 파일 경로 수정
    print("Dlib 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"Dlib 초기화 실패: {e}")
    exit()

# 눈 랜드마크 인덱스 설정
(lStart, lEnd) = (36, 41)  # 왼쪽 눈
(rStart, rEnd) = (42, 47)  # 오른쪽 눈

# EAR 임계값
EAR_THRESHOLD = 0.3  # EAR 값 기준
EAR_ALERT_THRESHOLD = 0.25  # 필요 시 더 낮게 조정

# DroidCam IP 주소 설정 (IP는 사용자가 DroidCam 설정에서 확인해야 함)
droidcam_url = "http://192.168.1.2:4747/video"  # DroidCam의 IP 스트림 URL

# DroidCam 비디오 캡처 객체 초기화
cap = cv2.VideoCapture(droidcam_url)
if not cap.isOpened():
    print("DroidCam 스트림을 열 수 없습니다.")
    exit()

print("DroidCam 스트림이 성공적으로 열렸습니다. 실시간 스트림을 시작합니다.")

# 실시간 비디오 스트림 처리
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        # 얼굴 랜드마크 예측
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # 눈 좌표 추출 및 EAR 계산
        leftEye = shape[lStart:lEnd + 1]
        rightEye = shape[rStart:rEnd + 1]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0  # 양쪽 눈의 평균 EAR

        # 눈 감지 마커 표시 (폴리곤)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  # 초록색
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)  # 초록색

        # EAR 값 기준으로 졸음 탐지
        if ear < EAR_THRESHOLD:
            cv2.putText(frame, "DROWSINESS DETECTION", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 빨간색
        else:
            cv2.putText(frame, "Awake", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # 초록색

        # EAR 값 화면에 표시
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 실시간 비디오 출력
    cv2.imshow("Drowsiness Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
