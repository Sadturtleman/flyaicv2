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

# 2. MediaPipe Face Landmarker 설정 (다중 얼굴 지원)
base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=4,  # 최대 4명까지 감지
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)
detector = vision.FaceLandmarker.create_from_options(options)

# 인덱스 정의
L_IRIS, R_IRIS = 468, 473
L_INNER, L_OUTER = 133, 33
R_INNER, R_OUTER = 362, 263

def get_detailed_gaze(landmarks):
    """양쪽 눈의 데이터를 종합하여 시선 방향(상하좌우/중심) 판별"""
    directions = []
    for iris_idx, inner_idx, outer_idx in [(L_IRIS, L_INNER, L_OUTER), (R_IRIS, R_INNER, R_OUTER)]:
        iris = landmarks[iris_idx]
        inner = landmarks[inner_idx]
        outer = landmarks[outer_idx]
        
        center_x = (inner.x + outer.x) / 2
        center_y = (inner.y + outer.y) / 2
        
        # 임계값 설정 (민감도 조절 가능)
        h_dir = ""
        if iris.x - center_x < -0.006: h_dir = "Right"
        elif iris.x - center_x > 0.006: h_dir = "Left"
        
        v_dir = ""
        if iris.y - center_y < -0.005: v_dir = "Up"
        elif iris.y - center_y > 0.005: v_dir = "Down"
        
        res = f"{h_dir} {v_dir}".strip()
        directions.append(res if res != "" else "Center")
    
    # 양쪽 눈 결과 중 빈도가 높은 것 반환
    return directions[0] if directions[0] == directions[1] else directions[0]

def get_face_bbox(landmarks, w, h, padding=0.2):
    """얼굴 크롭을 위한 바운딩 박스 계산"""
    x_pts = [l.x * w for l in landmarks]
    y_pts = [l.y * h for l in landmarks]
    x1, y1 = int(min(x_pts)), int(min(y_pts))
    x2, y2 = int(max(x_pts)), int(max(y_pts))
    
    pw, ph = (x2-x1) * padding, (y2-y1) * padding
    return max(0, int(x1-pw)), max(0, int(y1-ph)), min(w, int(x2+pw)), min(h, int(y2+ph))

cap = cv2.VideoCapture(0)
frame_count = 0
# 여러 명의 분석 결과를 저장할 리스트
faces_data = {} 

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_count += 1
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms == 0: timestamp_ms = frame_count * 33
    
    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    if detection_result.face_landmarks:
        for idx, landmarks in enumerate(detection_result.face_landmarks):
            # 1. 시선 판별
            gaze_dir = get_detailed_gaze(landmarks)
            
            # 2. 얼굴 속성 분석 (20프레임마다 각 얼굴별로 수행)
            if frame_count % 20 == 0 or idx not in faces_data:
                x1, y1, x2, y2 = get_face_bbox(landmarks, w, h)
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    try:
                        res = DeepFace.analyze(face_crop, actions=['age', 'gender', 'emotion'], 
                                               enforce_detection=False, silent=True)[0]
                        faces_data[idx] = res
                    except: pass
            
            # 3. 개별 얼굴 오버레이
            x1, y1, x2, y2 = get_face_bbox(landmarks, w, h, padding=0.05)
            # 얼굴 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 정보 텍스트 (얼굴 위에 표시)
            if idx in faces_data:
                d = faces_data[idx]
                info = f"{d['dominant_gender'][0]}, {d['age']}, {d['dominant_emotion']}"
                cv2.putText(frame, info, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Gaze: {gaze_dir}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 눈동자 포인트 시각화
            for iris_idx in [L_IRIS, R_IRIS]:
                pt = landmarks[iris_idx]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 0), -1)

    cv2.imshow('Multi-Face Gaze & Attribute Analysis', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()