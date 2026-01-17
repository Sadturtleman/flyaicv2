import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import threading

# DeepFace 결과를 저장할 전역 변수
analysis_results = None
is_analyzing = False

def analyze_face_async(frame_copy):
    global analysis_results, is_analyzing
    try:
        results = DeepFace.analyze(frame_copy, actions=['age', 'gender', 'emotion'], 
                                   enforce_detection=False, silent=True)
        analysis_results = results
    except Exception as e:
        print(f"Analysis error: {e}")
    finally:
        is_analyzing = False
 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# FPS 계산용
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. DeepFace 비동기 실행 (이전 분석이 끝났을 때만 새로운 분석 시작)
    if not is_analyzing:
        is_analyzing = True
        thread = threading.Thread(target=analyze_face_async, args=(frame.copy(),))
        thread.start()

    if analysis_results:
        for res in analysis_results:
            x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            info = f"{res['dominant_gender']}, {res['age']}, {res['dominant_emotion']}"
            cv2.putText(frame, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    
    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        left_iris = landmarks[468]
        gaze_x = left_iris.x
        if gaze_x < 0.45: gaze = "Right"
        elif gaze_x > 0.55: gaze = "Left"
        else: gaze = "Center"
        cv2.putText(frame, f"Gaze: {gaze}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 4. FPS 계산 및 표시
    curr_time = cv2.getTickCount()
    freq = cv2.getTickFrequency()
    fps = freq / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Fast Face Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()