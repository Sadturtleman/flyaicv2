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
    yaw_ratio = (nose.x - l_ear.x) / (r_ear.x - l_ear.x)
    
    # 2. Pitch (상하 회전) - 약식 계산
    mid_ear_y = (l_ear.y + r_ear.y) / 2
    pitch_val = nose.y - mid_ear_y
    
    return yaw_ratio, pitch_val

def get_detailed_gaze(landmarks):
    """
    [개선됨] 머리 방향(1순위) + 눈동자(2순위) 하이브리드 추적
    """
    global gaze_buffer
    
    # 1. 머리 방향(Head Pose) 분석
    yaw, pitch = get_head_pose_ratio(landmarks)
    
    # 2. 눈동자(Eye Gaze) 분석
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
    
    # 3. 최종 판단 로직
    final_h, final_v = "", ""

    # 좌우 판단
    if yaw < 0.40:    final_h = "Right"
    elif yaw > 0.60:  final_h = "Left"
    else:
        if avg_h_ratio < 0.44: final_h = "Right"
        elif avg_h_ratio > 0.56: final_h = "Left"
    
    # 상하 판단
    if pitch < -0.05:   final_v = "Up"
    elif pitch > 0.04:  final_v = "Down"
    else:
        if avg_v_ratio < 0.38: final_v = "Up"
        elif avg_v_ratio > 0.52: final_v = "Down"

    res = f"{final_h} {final_v}".strip()
    if res == "": res = "Center"
    
    # 4. 스무딩
    gaze_buffer.append(res)
    final_res = max(set(gaze_buffer), key=gaze_buffer.count)
    
    return final_res

# [핵심 수정] 각 ID별로 데이터와 분석 중인지 여부를 개별 저장
faces_status = {i: {'data': None, 'is_analyzing': False} for i in range(4)}

def draw_styled_panel(img, x, y, w, h, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# ==================== 감정 분석 함수 수정 ====================
def background_analysis(crop_img, idx):
    """[수정] 특정 인덱스의 얼굴만 개별적으로 분석하는 스레드 함수 - 감정 개선 적용"""
    global faces_status
    try:
        result = DeepFace.analyze(
            crop_img, 
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False, 
            silent=True,
            detector_backend='opencv'
        )[0]
        
        # 감정 평활화 적용
        raw_emotion = result['emotion']
        smoothed_emotion, smoothed_conf = get_smoothed_emotion(idx, raw_emotion)
        
        # 개선된 감정 정보 추가
        result['smoothed_emotion'] = smoothed_emotion
        result['smoothed_confidence'] = smoothed_conf
        
        # 해당 인덱스 슬롯에 결과 저장
        faces_status[idx]['data'] = result
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