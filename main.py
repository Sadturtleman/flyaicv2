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

# 인덱스 및 개별 상태 관리 변수
L_IRIS, R_IRIS = 468, 473
L_INNER, L_OUTER = 133, 33
L_TOP, L_BOTTOM = 159, 145
R_TOP, R_BOTTOM = 386, 374
R_INNER, R_OUTER = 362, 263

# [핵심 수정] 각 ID별로 데이터와 분석 중인지 여부를 개별 저장
# 예: faces_status[0] = {'data': {...}, 'is_analyzing': False}
faces_status = {i: {'data': None, 'is_analyzing': False} for i in range(4)}

def get_detailed_gaze(landmarks):
    """Ratio 기반 시선 추적"""
    avg_h_ratio, avg_v_ratio = 0, 0
    eyes_indices = [
        (L_IRIS, L_INNER, L_OUTER, L_TOP, L_BOTTOM),
        (R_IRIS, R_INNER, R_OUTER, R_TOP, R_BOTTOM)
    ]
    
    for iris_idx, inner_idx, outer_idx, top_idx, bottom_idx in eyes_indices:
        iris, inner, outer, top, bottom = landmarks[iris_idx], landmarks[inner_idx], landmarks[outer_idx], landmarks[top_idx], landmarks[bottom_idx]
        eye_width = ((outer.x - inner.x)**2 + (outer.y - inner.y)**2)**0.5
        if eye_width == 0: continue
        dist_to_inner = ((iris.x - inner.x)**2 + (iris.y - inner.y)**2)**0.5
        h_ratio = dist_to_inner / eye_width
        eye_height = ((bottom.x - top.x)**2 + (bottom.y - top.y)**2)**0.5
        if eye_height == 0: continue
        dist_to_top = ((iris.x - top.x)**2 + (iris.y - top.y)**2)**0.5
        v_ratio = dist_to_top / eye_height
        avg_h_ratio += h_ratio
        avg_v_ratio += v_ratio

    avg_h_ratio /= 2
    avg_v_ratio /= 2
    h_dir, v_dir = "", ""
    if avg_h_ratio < 0.42: h_dir = "Right"
    elif avg_h_ratio > 0.58: h_dir = "Left"
    if avg_v_ratio < 0.38: v_dir = "Up"
    elif avg_v_ratio > 0.55: v_dir = "Down"
    
    if h_dir == "" and v_dir == "": return "Center"
    return f"{h_dir} {v_dir}".strip()

def draw_styled_panel(img, x, y, w, h, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def background_analysis(crop_img, idx):
    """[수정] 특정 인덱스의 얼굴만 개별적으로 분석하는 스레드 함수"""
    global faces_status
    try:
        result = DeepFace.analyze(crop_img, actions=['age', 'gender', 'emotion'], 
                                  enforce_detection=False, silent=True)[0]
        # 해당 인덱스 슬롯에 결과 저장
        faces_status[idx]['data'] = result
    except Exception as e:
        print(f"Analysis Error for ID {idx+1}: {e}")
    finally:
        # 해당 인덱스 분석 종료 알림
        faces_status[idx]['is_analyzing'] = False

cap = cv2.VideoCapture(0)
frame_count = 0
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    
    frame_count += 1
    h, w, _ = frame.shape
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms == 0: timestamp_ms = frame_count * 33
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    # 상단 대시보드
    num_detected = len(result.face_landmarks) if result.face_landmarks else 0
    frame = draw_styled_panel(frame, 10, 10, 320, 45 + (num_detected * 85))
    cv2.putText(frame, "INDIVIDUAL LIVE MONITOR", (20, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"FPS: {int(fps)}", (350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if result.face_landmarks:
        for idx, landmarks in enumerate(result.face_landmarks):
            if idx >= 4: break # 설정한 num_faces=4 까지만 처리
            
            gaze = get_detailed_gaze(landmarks)
            x_pts = [l.x * w for l in landmarks]
            y_pts = [l.y * h for l in landmarks]
            x1, y1, x2, y2 = int(min(x_pts)), int(min(y_pts)), int(max(x_pts)), int(max(y_pts))

            # [수정] 개별 분석 로직: 이 ID(idx)가 현재 분석 중이 아닐 때만 25프레임마다 실행
            if frame_count % 25 == 0 and not faces_status[idx]['is_analyzing']:
                margin = int((x2-x1)*0.2)
                crop = frame[max(0, y1-margin):min(h, y2+margin), max(0, x1-margin):min(w, x2+margin)]
                if crop.size > 0:
                    faces_status[idx]['is_analyzing'] = True
                    # 해당 idx 정보를 들고 스레드 시작
                    threading.Thread(target=background_analysis, args=(crop.copy(), idx), daemon=True).start()

            # UI 표시 로직
            panel_y = 65 + (idx * 85)
            cv2.line(frame, (20, panel_y-5), (300, panel_y-5), (80, 80, 80), 1)
            
            # 각 ID별 저장된 데이터 가져와서 표시
            if faces_status[idx]['data']:
                d = faces_status[idx]['data']
                gender = "M" if d['dominant_gender'] == 'Man' else "W"
                info = f"ID:{idx+1} | {gender} | {d['age']}s"
                emo = f"Emotion: {d['dominant_emotion'].upper()}"
                cv2.putText(frame, info, (25, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, emo, (25, panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(frame, f"ID:{idx+1} Initializing...", (25, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.putText(frame, f"Gaze: {gaze}", (25, panel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 얼굴 박스 및 번호 (분석 중일 때는 노란색으로 표시)
            box_color = (0, 255, 255) if faces_status[idx]['is_analyzing'] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
            cv2.putText(frame, f"FACE {idx+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            
            # 눈동자 포인트
            for i_idx in [L_IRIS, R_IRIS]:
                p = landmarks[i_idx]
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (0, 255, 255), -1)

    cv2.imshow('Face Dashboard (Individual Analysis)', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()