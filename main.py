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
from collections import deque
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

# 눈 관련
L_IRIS, R_IRIS = 468, 473
L_INNER, L_OUTER = 133, 33
R_INNER, R_OUTER = 362, 263
L_TOP, L_BOTTOM = 159, 145
R_TOP, R_BOTTOM = 386, 374

# 머리 방향(Head Pose) 관련 (코 끝, 양쪽 귀 근처)
NOSE_TIP = 1
L_EAR = 234  # 왼쪽 광대/귀 쪽
R_EAR = 454  # 오른쪽 광대/귀 쪽

# [안정화] 최근 5프레임의 데이터를 저장할 버퍼 (떨림 방지용)
gaze_buffer = deque(maxlen=5)

def get_head_pose_ratio(landmarks):
    """
    얼굴이 좌우/상하로 얼마나 돌아갔는지 비율로 계산
    반환값: (yaw_ratio, pitch_ratio)
    """
    nose = landmarks[NOSE_TIP]
    l_ear = landmarks[L_EAR]
    r_ear = landmarks[R_EAR]
    
    # 1. Yaw (좌우 회전) 계산
    # 양쪽 귀 사이의 거리
    face_width = ((r_ear.x - l_ear.x)**2 + (r_ear.y - l_ear.y)**2)**0.5
    if face_width == 0: return 0.5, 0.5
    
    # 코가 양쪽 귀 중앙에서 얼마나 벗어났는지 확인
    # 0.0(Left) ~ 0.5(Center) ~ 1.0(Right)
    # (이미지 기준 Left/Right이므로 실제 사용자 기준과는 반대일 수 있음 -> 아래 로직에서 보정)
    yaw_ratio = (nose.x - l_ear.x) / (r_ear.x - l_ear.x)
    
    # 2. Pitch (상하 회전) - 약식 계산
    # 코와 귀의 Y축 관계 이용 (고개 숙이면 코가 귀보다 내려가거나 올라감)
    mid_ear_y = (l_ear.y + r_ear.y) / 2
    # 값이 클수록(양수) 코가 귀보다 아래 -> 고개 숙임(Down)
    # 값이 작을수록(음수) 코가 귀보다 위 -> 고개 듦(Up)
    pitch_val = nose.y - mid_ear_y
    
    return yaw_ratio, pitch_val

def get_detailed_gaze(landmarks):
    """
    [개선됨] 머리 방향(1순위) + 눈동자(2순위) 하이브리드 추적
    """
    global gaze_buffer
    
    # 1. 머리 방향(Head Pose) 분석
    yaw, pitch = get_head_pose_ratio(landmarks)
    
    # 2. 눈동자(Eye Gaze) 분석 (기존 로직 유지)
    avg_h_ratio, avg_v_ratio = 0, 0
    eyes_indices = [
        (L_IRIS, L_INNER, L_OUTER, L_TOP, L_BOTTOM),
        (R_IRIS, R_INNER, R_OUTER, R_TOP, R_BOTTOM)
    ]
    
    valid_eyes = 0
    for iris_idx, inner_idx, outer_idx, top_idx, bottom_idx in eyes_indices:
        iris = landmarks[iris_idx]
        inner = landmarks[inner_idx]
        outer = landmarks[outer_idx]
        top = landmarks[top_idx]
        bottom = landmarks[bottom_idx]

        ew = ((outer.x - inner.x)**2 + (outer.y - inner.y)**2)**0.5
        eh = ((bottom.x - top.x)**2 + (bottom.y - top.y)**2)**0.5
        if ew == 0 or eh == 0: continue
        
        avg_h_ratio += ((iris.x - inner.x)**2 + (iris.y - inner.y)**2)**0.5 / ew
        avg_v_ratio += ((iris.x - top.x)**2 + (iris.y - top.y)**2)**0.5 / eh
        valid_eyes += 1
        
    if valid_eyes > 0:
        avg_h_ratio /= valid_eyes
        avg_v_ratio /= valid_eyes
    
    # 3. [최종 판단 로직] Head Pose 우선 적용
    final_h, final_v = "", ""

    # (1) 좌우 판단: 머리 회전이 크면 눈동자 무시하고 머리 방향 따름
    # Head Yaw 기준값 (0.5가 정면)
    # 0.4 미만이면 고개 오른쪽 돌림 / 0.6 초과면 고개 왼쪽 돌림 (값은 튜닝 필요)
    if yaw < 0.40:    final_h = "Right"  # 고개를 우측으로 돌림
    elif yaw > 0.60:  final_h = "Left"   # 고개를 좌측으로 돌림
    else:
        # 고개가 정면(0.40 ~ 0.60)일 때만 눈동자 확인
        if avg_h_ratio < 0.44: final_h = "Right"
        elif avg_h_ratio > 0.56: final_h = "Left"
    
    # (2) 상하 판단: 머리 숙임 우선
    # pitch가 양수면 코가 아래로(Down), 음수면 위로(Up)
    # Head Pitch 기준값 (사용자마다 다를 수 있음)
    if pitch < -0.05:   final_v = "Up"     # 고개 듦
    elif pitch > 0.04:  final_v = "Down"   # 고개 숙임
    else:
        # 고개가 정면일 때 눈동자 확인
        if avg_v_ratio < 0.38: final_v = "Up"
        elif avg_v_ratio > 0.52: final_v = "Down"

    # 결과 문자열 조합
    res = f"{final_h} {final_v}".strip()
    if res == "": res = "Center"
    
    # 4. [스무딩] 결과 튀는 현상 방지 (최빈값 사용)
    gaze_buffer.append(res)
    # 버퍼에서 가장 많이 등장한 값 선택 (Voting)
    final_res = max(set(gaze_buffer), key=gaze_buffer.count)
    
    # 디버깅용 수치 반환 (튜닝할 때 이 값을 보세요)
    debug_str = f"Head:{yaw:.2f} Eye:{avg_h_ratio:.2f}"
    
    return final_res, debug_str
# [핵심 수정] 각 ID별로 데이터와 분석 중인지 여부를 개별 저장
# 예: faces_status[0] = {'data': {...}, 'is_analyzing': False}
faces_status = {i: {'data': None, 'is_analyzing': False} for i in range(4)}

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