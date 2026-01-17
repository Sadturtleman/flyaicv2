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

# ==================== 감정 분석 개선 시작 ====================

# 감정 히스토리 저장 (각 얼굴별)
emotion_histories = {}
HISTORY_SIZE = 10  # 최근 10개 프레임 저장

# 감정 가중치 - 5가지 사용
EMOTION_WEIGHTS = {
    'angry': 1.2,      # 부정 감정 대표
    'happy': 1.0,      # 긍정 감정 대표
    'sad': 1.1,        # 슬픈 감정
    'surprise': 0.95,  # 놀람 (약간 낮게)
    'neutral': 0.7     # 무표정
}

# 신뢰도 임계값
CONFIDENCE_THRESHOLD = 0.25

def apply_emotion_weights(emotion_scores):
    """감정 점수에 가중치 적용"""
    weighted = {}
    for emotion, score in emotion_scores.items():
        weight = EMOTION_WEIGHTS.get(emotion, 1.0)
        weighted[emotion] = score * weight
    return weighted

def get_smoothed_emotion(face_id, raw_emotion_data):
    """
    시계열 평활화로 안정적인 감정 반환 (5가지 감정으로 단순화)
    - fear, disgust는 angry로 통합
    
    Args:
        face_id: 얼굴 ID
        raw_emotion_data: DeepFace 원본 결과의 emotion dict (7가지)
    
    Returns:
        tuple: (smoothed_emotion, confidence) - 5가지 중 하나
    """
    # 히스토리 초기화
    if face_id not in emotion_histories:
        emotion_histories[face_id] = deque(maxlen=HISTORY_SIZE)
    
    # ========== 감정 단순화: 7가지 → 5가지 ==========
    simplified_emotion = {
        'angry': 0,
        'happy': 0,
        'sad': 0,
        'surprise': 0,
        'neutral': 0
    }
    
    # fear와 disgust를 angry로 통합 (80% 비율로 합산)
    simplified_emotion['angry'] = (
        raw_emotion_data.get('angry', 0) +
        raw_emotion_data.get('fear', 0) * 0.8 +
        raw_emotion_data.get('disgust', 0) * 0.8
    )
    
    # 나머지 감정은 그대로 사용
    simplified_emotion['happy'] = raw_emotion_data.get('happy', 0)
    simplified_emotion['sad'] = raw_emotion_data.get('sad', 0)
    simplified_emotion['surprise'] = raw_emotion_data.get('surprise', 0)
    simplified_emotion['neutral'] = raw_emotion_data.get('neutral', 0)
    # ================================================
    
    # 가중치 적용 (단순화된 5가지 감정에 적용)
    weighted_scores = apply_emotion_weights(simplified_emotion)
    
    # 현재 감정의 주요 감정과 점수
    current_emotion = max(weighted_scores, key=weighted_scores.get)
    current_score = weighted_scores[current_emotion] / 100.0
    
    # 신뢰도가 너무 낮으면 히스토리에 추가하지 않음
    if current_score >= CONFIDENCE_THRESHOLD:
        emotion_histories[face_id].append(weighted_scores)
    
    # 히스토리가 충분하지 않으면 현재 값 사용
    if len(emotion_histories[face_id]) < 3:
        return current_emotion, weighted_scores[current_emotion]
    
    # 가중 평균 계산 (최근일수록 가중치 높음)
    emotion_sums = {}
    total_weight = 0
    
    for i, scores in enumerate(emotion_histories[face_id]):
        weight = (i + 1) / len(emotion_histories[face_id])  # 최근일수록 가중치 증가
        total_weight += weight
        
        for emotion, score in scores.items():
            emotion_sums[emotion] = emotion_sums.get(emotion, 0) + (score * weight)
    
    # 가중 평균
    emotion_avgs = {k: v / total_weight for k, v in emotion_sums.items()}
    
    # 최종 감정 결정
    smoothed_emotion = max(emotion_avgs, key=emotion_avgs.get)
    smoothed_confidence = emotion_avgs[smoothed_emotion]
    
    return smoothed_emotion, smoothed_confidence

# ==================== 감정 분석 개선 끝 ====================

def get_head_pose_ratio(landmarks):
    nose = landmarks[NOSE_TIP]
    l_ear = landmarks[L_EAR]
    r_ear = landmarks[R_EAR]
    
    # Yaw (좌우)
    face_width = ((r_ear.x - l_ear.x)**2 + (r_ear.y - l_ear.y)**2)**0.5
    if face_width == 0: return 0.5, 0.5
    yaw_ratio = (nose.x - l_ear.x) / (r_ear.x - l_ear.x) # 0.5가 정면
    
    # Pitch (상하)
    mid_ear_y = (l_ear.y + r_ear.y) / 2
    pitch_val = nose.y - mid_ear_y # 양수=Down, 음수=Up
    
    return yaw_ratio, pitch_val

def get_detailed_gaze(landmarks):
    global gaze_buffer
    
    # 1. 머리 방향 먼저 계산
    yaw, pitch = get_head_pose_ratio(landmarks)
    
    # [눈 선택 로직] 고개를 많이 돌렸을 때 반대쪽 눈 무시 (기존 동일)
    target_eyes = []
    
    if yaw > 0.65:   # Left Turn -> Left Eye Used
        target_eyes = [(L_IRIS, L_INNER, L_OUTER, L_TOP, L_BOTTOM)]
    elif yaw < 0.35: # Right Turn -> Right Eye Used
        target_eyes = [(R_IRIS, R_INNER, R_OUTER, R_TOP, R_BOTTOM)]
    else:            # Front -> Both Eyes
        target_eyes = [
            (L_IRIS, L_INNER, L_OUTER, L_TOP, L_BOTTOM),
            (R_IRIS, R_INNER, R_OUTER, R_TOP, R_BOTTOM)
        ]

    avg_h_ratio, avg_v_ratio = 0, 0
    valid_eyes = 0
    
    for iris_idx, inner_idx, outer_idx, top_idx, bottom_idx in target_eyes:
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
    
    # ### 빨간박스 ######################################################################
    # [수정됨] Center 범위를 넓히기 위해 임계값(Threshold) 완화
    # 목표: 정면을 보고 있을 때 'Left Down' 같은 오탐지 줄이기
    
    final_h = ""
    final_v = ""

    # (A) 수평(Horizontal) 판단
    # Head Yaw 범위 확장: 0.35/0.65 -> 0.32/0.68 (머리를 좀 더 돌려야 인식)
    if yaw < 0.32:      final_h = "Right" 
    elif yaw > 0.68:    final_h = "Left"
    else:
        # Eye Ratio 범위 확장: 0.46/0.54 -> 0.43/0.57
        # 이제 0.43 ~ 0.57 사이는 전부 Center로 간주함 (안전지대 확대)
        if avg_h_ratio < 0.43: final_h = "Right"
        elif avg_h_ratio > 0.57: final_h = "Left"

    # (B) 수직(Vertical) 판단
    # Head Pitch 범위 확장: -0.06/0.05 -> -0.08/0.08
    if pitch < -0.08:     final_v = "Up"
    elif pitch > 0.08:    final_v = "Down"
    else:
        # Eye Ratio 범위 확장
        # Up: 0.42 이하 (유지 - 위쪽 보기는 원래 힘듦)
        # Down: 0.53 -> 0.56 (아래쪽 보기는 너무 민감해서 기준을 높임)
        if avg_v_ratio < 0.42: final_v = "Up"
        elif avg_v_ratio > 0.56: final_v = "Down" # 0.56 넘어야 Down

    # 결과 문자열 조합
    res_parts = []
    if final_h: res_parts.append(final_h)
    if final_v: res_parts.append(final_v)
    
    res = " ".join(res_parts)
    if res == "": res = "Center"
    # ### 빨간박스 ######################################################################
    
    # 스무딩
    gaze_buffer.append(res)
    final_res = max(set(gaze_buffer), key=gaze_buffer.count)
    
    debug_str = f"Y:{yaw:.2f} P:{pitch:.2f} EyeV:{avg_v_ratio:.2f}"
    
    return final_res, debug_str

# [핵심 수정] 각 ID별로 데이터와 분석 중인지 여부를 개별 저장
faces_status = {i: {'data': None, 'is_analyzing': False} for i in range(4)}

def draw_styled_panel(img, x, y, w, h, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# ==================== 감정 분석 함수 수정 ====================
def background_analysis(crop_img, idx):
    """특정 인덱스의 얼굴만 개별적으로 분석 + (나이/성별) 스무딩/보정, 감정은 원본 유지"""
    global faces_status
    try:
        # --- Preprocess: 안정화 (과노화/미검출 줄임) ---
        face_bgr = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_LINEAR)
        face_bgr = cv2.GaussianBlur(face_bgr, (3, 3), 0)
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # --- Analyze (crop 했으니 detector는 skip이 흔들림 적음) ---
        res = DeepFace.analyze(
            face_rgb,
            actions=['age', 'gender', 'emotion'],
            detector_backend='skip',
            enforce_detection=False,
            silent=True
        )[0]

        prev = faces_status[idx]['data'] or {}
        alpha = 0.2  # 0.1이면 더 부드럽게(느림), 0.3이면 더 빠르게(튐 증가)

        # =========================
        # AGE: 보정 + 스무딩
        # =========================
        if 'age' in res:
            age_cal = float(res['age']) - 4

            # 말도 안되게 튀는 값 제한
            age_cal = max(5, min(80, age_cal))

            if isinstance(prev.get('age'), (int, float)):
                age_cal = (1 - alpha) * float(prev['age']) + alpha * age_cal

            res['age'] = int(round(age_cal))

        # =========================
        # GENDER: dict 기반이면 EMA로 안정화
        # =========================
        if isinstance(res.get('gender'), dict):
            ema_g = prev.get('_ema_gender', {'Man': None, 'Woman': None})
            for k in ['Man', 'Woman']:
                if k in res['gender']:
                    v = float(res['gender'][k])
                    ema_g[k] = v if ema_g.get(k) is None else (1 - alpha) * float(ema_g[k]) + alpha * v
            res['_ema_gender'] = ema_g

            man = ema_g.get('Man') or 0.0
            wom = ema_g.get('Woman') or 0.0
            res['dominant_gender'] = 'Man' if man >= wom else 'Woman'
        else:
            if isinstance(res.get('dominant_gender'), str):
                res['dominant_gender'] = res['dominant_gender']

        # --- 기존 값 유지하면서 업데이트 (EMA 상태도 유지됨) ---
        merged = dict(prev)
        merged.update(res)
        faces_status[idx]['data'] = merged

    except Exception as e:
        print(f"Analysis Error for ID {idx+1}: {e}")
    finally:
        faces_status[idx]['is_analyzing'] = False
# ==================== 감정 분석 함수 수정 끝 ====================

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
            if idx >= 4: break
            
            gaze = get_detailed_gaze(landmarks)
            x_pts = [l.x * w for l in landmarks]
            y_pts = [l.y * h for l in landmarks]
            x1, y1, x2, y2 = int(min(x_pts)), int(min(y_pts)), int(max(x_pts)), int(max(y_pts))

            # ==================== 분석 주기 수정 (20프레임마다) ====================
            if (frame_count % 20 == 0 or frame_count < 5) and not faces_status[idx]['is_analyzing']:
                margin = int((x2-x1)*0.2)
                crop = frame[max(0, y1-margin):min(h, y2+margin), max(0, x1-margin):min(w, x2+margin)]
                if crop.size > 0:
                    faces_status[idx]['is_analyzing'] = True
                    threading.Thread(target=background_analysis, args=(crop.copy(), idx), daemon=True).start()
            # ====================================================================

            # UI 표시 로직
            panel_y = 65 + (idx * 85)
            cv2.line(frame, (20, panel_y-5), (300, panel_y-5), (80, 80, 80), 1)
            
            # ==================== 감정 표시 수정 ====================
            if faces_status[idx]['data']:
                d = faces_status[idx]['data']
                gender = "M" if d['dominant_gender'] == 'Man' else "W"
                info = f"ID:{idx+1} | {gender} | {d['age']}s"
                
                # 평활화된 감정 사용 (있으면)
                if 'smoothed_emotion' in d:
                    display_emotion = d['smoothed_emotion']
                    display_conf = d['smoothed_confidence']
                    emo = f"Emotion: {display_emotion.upper()} ({display_conf:.0f}%)"
                else:
                    emo = f"Emotion: {d['dominant_emotion'].upper()}"
                
                cv2.putText(frame, info, (25, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, emo, (25, panel_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(frame, f"ID:{idx+1} Initializing...", (25, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            # ===================================================

            cv2.putText(frame, f"Gaze: {gaze}", (25, panel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 얼굴 박스 및 번호
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
