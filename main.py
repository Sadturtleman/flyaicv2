import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace

# 1. 모델 파일 확인 및 다운로드
MODEL_FILE = 'face_landmarker.task'
if not os.path.exists(MODEL_FILE):
    print("모델 다운로드 중...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_FILE)

# 2. MediaPipe Face Landmarker 설정 (VIDEO 모드 최적화)
base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # 실시간 비디오 모드
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)
detector = vision.FaceLandmarker.create_from_options(options)

# 시선 계산을 위한 눈 주변 랜드마크 인덱스
L_EYE_IRIS = 468
R_EYE_IRIS = 473
L_EYE_OUTER = 33
L_EYE_INNER = 133
R_EYE_INNER = 362
R_EYE_OUTER = 263

def detect_gaze(iris, inner, outer):
    """눈의 안쪽/바깥쪽 끝점을 기준으로 홍채의 위치를 파악하여 방향 반환"""
    # 눈의 중심점 계산
    eye_center_x = (inner.x + outer.x) / 2
    eye_center_y = (inner.y + outer.y) / 2
    
    # 홍채와 중심점의 거리 차이 (임계값 설정)
    dx = iris.x - eye_center_x
    dy = iris.y - eye_center_y
    
    horizontal = "Center"
    if dx < -0.005: horizontal = "Right" # 화면 기준
    elif dx > 0.005: horizontal = "Left"
    
    vertical = "Center"
    if dy < -0.004: vertical = "Up"
    elif dy > 0.004: vertical = "Down"
    
    return horizontal, vertical

cap = cv2.VideoCapture(0)
frame_count = 0
analysis_results = None
gaze_text = "Gaze: Calculating..."

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_count += 1
    # VIDEO 모드에서는 매 프레임마다 증가하는 타임스탬프가 필요함
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms == 0: timestamp_ms = frame_count * 33 # 30fps 기준 보정
    
    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    # [기능 1, 2] 얼굴 및 랜드마크 감지
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]

        # [기능 4] 시선 방향 추적 (좌/우/상/하)
        l_h, l_v = detect_gaze(landmarks[L_EYE_IRIS], landmarks[L_EYE_INNER], landmarks[L_EYE_OUTER])
        r_h, r_v = detect_gaze(landmarks[R_EYE_IRIS], landmarks[R_EYE_INNER], landmarks[R_EYE_OUTER])
        
        # 양쪽 눈의 평균적인 방향 결정
        gaze_text = f"Gaze: {l_h} / {l_v}"

        # [기능 3] 얼굴 속성 분류 (15프레임마다 DeepFace 실행)
        if frame_count % 15 == 0 or analysis_results is None:
            try:
                # 얼굴 부분만 크롭하여 분석하면 더 정확하지만, 여기선 전체 프레임 사용
                analysis_results = DeepFace.analyze(
                    frame, actions=['age', 'gender', 'emotion'],
                    enforce_detection=False, silent=True
                )[0]
            except: pass

        # [기능 5] 오버레이 시각화
        # 시선 표시 (눈 위에 점 및 텍스트)
        for idx in [L_EYE_IRIS, R_EYE_IRIS]:
            pt = landmarks[idx]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, (0, 255, 0), -1)
        
        # 속성 정보 텍스트 오버레이
        if analysis_results:
            attr_text = f"{analysis_results['dominant_gender']}, {analysis_results['age']}s, {analysis_results['dominant_emotion']}"
            cv2.rectangle(frame, (10, 10), (450, 90), (0, 0, 0), -1) # 배경 박스
            cv2.putText(frame, attr_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, gaze_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Face AI Comprehensive Analysis', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()