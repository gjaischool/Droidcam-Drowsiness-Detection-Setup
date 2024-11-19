import cv2
import dlib
import numpy as np
import pygame
import joblib
from scipy.spatial import distance as dist
from scipy.signal import find_peaks

# Pygame 초기화
pygame.mixer.init()

# 상황별 알림음 파일 경로 설정
drowsiness_sound_path = 'siren-alert-96052.mp3'
head_tilt_sound_path = 'warning-alert-this-is-not-a-test-141753.mp3'
eye_closed_sound_path = 'eye-closed-alert.mp3'

# 심박수 측정을 위한 초기화
heart_rate_buffer = []
buffer_size = 150
fps = 30

# 초기 상태 설정
initial_ear = None
initial_mar = None

# 임계값 설정
EAR_THRESHOLD = 0.1
PITCH_THRESHOLD = 15
FRAME_THRESHOLD = int(fps * 2)
eye_blink_counter = 0
head_tilt_counter = 0
eye_open_counter = 0
alarm_active = False

# EAR 계산 함수 (눈 감김 비율)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# MAR 계산 함수 (하품 비율)
def calculate_mar(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# 오일러 각도 계산 함수 (Yaw, Pitch, Roll)
def get_euler_angles(shape, frame):
    image_points = np.array([
        shape[30],  # 코 끝 (nose tip)
        shape[8],   # 턱 끝 (chin)
        shape[36],  # 왼쪽 눈 좌측 끝 (left eye left corner)
        shape[45],  # 오른쪽 눈 우측 끝 (right eye right corner)
        shape[48],  # 왼쪽 입 끝 (left mouth corner)
        shape[54]   # 오른쪽 입 끝 (right mouth corner)
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             
        (0.0, -330.0, -65.0),        
        (-225.0, 170.0, -135.0),     
        (225.0, 170.0, -135.0),      
        (-150.0, -150.0, -125.0),    
        (150.0, -150.0, -125.0)      
    ])

    focal_length = frame.shape[1]
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    yaw, pitch, roll = angles
    return yaw, pitch, roll

# 체온 추정을 위한 함수
def estimate_skin_temperature(frame, forehead_region):
    roi = frame[forehead_region[1]:forehead_region[1]+forehead_region[3],
                forehead_region[0]:forehead_region[0]+forehead_region[2]]
    mean_val = np.mean(roi)
    estimated_temp = 30 + (mean_val / 255.0) * 10
    return estimated_temp

# 얼굴에서 심박수를 측정하는 함수
def measure_heart_rate(frame, forehead_region):
    roi = frame[forehead_region[1]:forehead_region[1]+forehead_region[3],
                forehead_region[0]:forehead_region[0]+forehead_region[2]]
    mean_val = np.mean(roi)

    heart_rate_buffer.append(mean_val)
    if len(heart_rate_buffer) > buffer_size:
        heart_rate_buffer.pop(0)

    if len(heart_rate_buffer) == buffer_size:
        signal = np.diff(heart_rate_buffer)
        peaks, _ = find_peaks(signal, distance=15)
        heart_rate = len(peaks) * (fps / buffer_size) * 60
        return heart_rate
    return None

# 안경 감지 모델 로드
try:
    glasses_model_1 = joblib.load('path_to_lfw_model.pkl') # 정확한 경로로 설정
    glasses_model_2 = joblib.load('path_to_celeba_model.pkl') # 정확한 경로로 설정
except FileNotFoundError:
    print("모델 파일이 존재하지 않습니다. 올바른 경로를 설정하세요.")
    exit()

# 특징 추출 함수
def extract_features(image):
    features = np.mean(image, axis=(0, 1)).flatten()
    return features

# 안경 감지 함수
def detect_glasses(image):
    features = extract_features(image)
    prediction_1 = glasses_model_1.predict([features])[0]
    prediction_2 = glasses_model_2.predict([features])[0]
    return prediction_1 == 1 or prediction_2 == 1

# 얼굴 탐지기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/USER/Desktop/shape_predictor_68_face_landmarks.dat')

# 눈과 입 랜드마크 인덱스
(lStart, lEnd) = (36, 41)
(rStart, rEnd) = (42, 47)
(mStart, mEnd) = (48, 67)
nose_idx = 30

# 비디오 캡처 시작
cap = cv2.VideoCapture(0)

# 초기 상태 설정 단계
initial_set = False
while not initial_set:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        leftEye = shape[lStart:lEnd + 1]
        rightEye = shape[rStart:rEnd + 1]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mouth = shape[mStart:mEnd + 1]
        mar = calculate_mar(mouth)

        initial_ear = ear
        initial_mar = mar
        initial_set = True
        print("초기 상태가 설정되었습니다.")

    cv2.putText(frame, "Setting Initial State. Please Look at the Camera.", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Drowsiness Detection - Initial Setup", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 실시간 탐지 시작
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        leftEye = shape[lStart:lEnd + 1]
        rightEye = shape[rStart:rEnd + 1]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # 안경 착용 여부 감지
        is_glasses = detect_glasses(frame)
        adjusted_ear_threshold = EAR_THRESHOLD * 1.2 if is_glasses else EAR_THRESHOLD

        mouth = shape[mStart:mEnd + 1]
        mar = calculate_mar(mouth)

        yaw, pitch, roll = get_euler_angles(shape, frame)

        forehead_region = (face.left(), face.top(), face.width(), face.height() // 4)
        heart_rate = measure_heart_rate(frame, forehead_region)
        skin_temperature = estimate_skin_temperature(frame, forehead_region)

        # 졸음 및 고개 기울임 감지
        if ear < adjusted_ear_threshold:
            eye_blink_counter += 1
            eye_open_counter = 0
        else:
            eye_blink_counter = 0
            eye_open_counter += 1

        if abs(pitch) > PITCH_THRESHOLD:
            head_tilt_counter += 1
        else:
            head_tilt_counter = 0

        if eye_blink_counter >= FRAME_THRESHOLD and not alarm_active:
            pygame.mixer.music.load(eye_closed_sound_path)
            pygame.mixer.music.play()
            alarm_active = True

        if alarm_active and (eye_open_counter >= int(fps * 1) or head_tilt_counter >= int(fps * 2)):
            pygame.mixer.music.stop()
            alarm_active = False

        for i, (x, y) in enumerate(shape):
            if i in range(lStart, lEnd + 1) or i in range(rStart, rEnd + 1):
                color = (255, 0, 0)
            elif i in range(mStart, mEnd + 1):
                color = (0, 255, 0)
            elif i == nose_idx:
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)
            cv2.circle(frame, (x, y), 2, color, -1)

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if heart_rate:
            cv2.putText(frame, f"Heart Rate: {int(heart_rate)} BPM", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Estimated Temp: {skin_temperature:.2f} C", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
