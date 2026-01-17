import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace

# 1. 모델 설정 및 다운로드
MODEL_FILE = 'face_landmarker.task'
if not os.path.exists(MODEL_FILE):
    print("모델 다운로드 중...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_FILE)

base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=4,
    output_face_blendshapes=True
)
detector = vision.FaceLandmarker.create_from_options(options)

# 인덱스 및 전역 변수 설정
L_IRIS, R_IRIS = 468, 473
L_INNER, L_OUTER = 133, 33
R_INNER, R_OUTER = 362, 263

faces_data = {}  # 분석 결과 저장소
is_analyzing = False  # 현재 분석 중인지 확인하는 플래그

def get_detailed_gaze(landmarks):
    """상하좌우 중심 시선 판별"""
    res_h, res_v = "", ""
    for iris_idx, inner_idx, outer_idx in [(L_IRIS, L_INNER, L_OUTER), (R_IRIS, R_INNER, R_OUTER)]:
        iris, inner, outer = landmarks[iris_idx], landmarks[inner_idx], landmarks[outer_idx]
        dx, dy = iris.x - (inner.x + outer.x)/2, iris.y - (inner.y + outer.y)/2
        if dx < -0.006: res_h = "Right"
        elif dx > 0.006: res_h = "Left"
        if dy < -0.005: res_v = "Up"
        elif dy > 0.005: res_v = "Down"
    
    h_part = res_h if res_h else ""
    v_part = res_v if res_v else ""
    final = f"{h_part} {v_part}".strip()
    return final if final else "Center"

def draw_styled_panel(img, x, y, w, h, alpha=0.6):
    """정보창을 위한 반투명 패널"""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def background_analysis(crop_img, idx):
    """DeepFace 분석을 수행하는 백그라운드 함수"""
    global faces_data, is_analyzing
    try:
        # 실행 시간을 잡아먹는 주범 (DeepFace 연산)
        result = DeepFace.analyze(crop_img, actions=['age', 'gender', 'emotion'], 
                                  enforce_detection=False, silent=True)[0]
        faces_data[idx] = result
    except Exception as e:
        print(f"Analysis Error: {e}")
    finally:
        is_analyzing = False # 분석 완료 후 플래그 해제

cap = cv2.VideoCapture(0)
frame_count = 0
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # FPS 계산
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    
    frame_count += 1
    h, w, _ = frame.shape
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms == 0: timestamp_ms = frame_count * 33
    
    # 1. MediaPipe 시선 추적 (매우 빠름)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    num_detected = len(result.face_landmarks) if result.face_landmarks else 0
    frame = draw_styled_panel(frame, 10, 10, 320, 40 + (num_detected * 85) if num_detected > 0 else 40)
    cv2.putText(frame, "LIVE MONITOR", (20, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps)}", (240, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    if result.face_landmarks:
        for idx, landmarks in enumerate(result.face_landmarks):
            gaze = get_detailed_gaze(landmarks)
            x_pts = [l.x * w for l in landmarks]
            y_pts = [l.y * h for l in landmarks]
            x1, y1, x2, y2 = int(min(x_pts)), int(min(y_pts)), int(max(x_pts)), int(max(y_pts))

            # 2. 비동기 분석 실행 (25프레임마다 + 현재 분석 중이 아닐 때만)
            if frame_count % 25 == 0 and not is_analyzing:
                margin = int((x2-x1)*0.2)
                crop = frame[max(0, y1-margin):min(h, y2+margin), max(0, x1-margin):min(w, x2+margin)]
                if crop.size > 0:
                    is_analyzing = True
                    # 별도 스레드에서 DeepFace 실행
                    threading.Thread(target=background_analysis, args=(crop.copy(), idx), daemon=True).start()

            # 3. 화면 UI 표시 (기존 분석 결과가 있으면 표시)
            panel_y = 60 + (idx * 85)
            cv2.line(frame, (20, panel_y-5), (300, panel_y-5), (50, 50, 50), 1)
            
            if idx in faces_data:
                d = faces_data[idx]
                attr = f"ID:{idx+1} | {d['dominant_gender'][0]} | {d['age']}s"
                emo = f"Emotion: {d['dominant_emotion'].upper()}"
                cv2.putText(frame, attr, (25, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, emo, (25, panel_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            
            cv2.putText(frame, f"Gaze: {gaze}", (25, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, str(idx + 1), (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            for i_idx in [L_IRIS, R_IRIS]:
                p = landmarks[i_idx]
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (0, 255, 255), -1)

    cv2.imshow('Face Dashboard (Threaded)', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()