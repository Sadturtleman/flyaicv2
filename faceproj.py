import cv2
import numpy as np
from deepface import DeepFace
import threading

# DeepFace 결과 저장
analysis_results = None
is_analyzing = False
frame_count = 0

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

# OpenCV 얼굴/눈 검출
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1

    # DeepFace 분석 (30프레임마다)
    if frame_count % 30 == 0 and not is_analyzing:
        is_analyzing = True
        thread = threading.Thread(target=analyze_face_async, args=(frame.copy(),))
        thread.start()

    # DeepFace 결과 표시
    if analysis_results:
        for res in analysis_results:
            x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            info = f"{res['dominant_gender']}, {res['age']}, {res['dominant_emotion']}"
            cv2.putText(frame, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 간단한 시선 추적
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            # 왼쪽 눈, 오른쪽 눈 정렬
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            eye_centers = [(x + ex + ew//2, y + ey + eh//2) for (ex, ey, ew, eh) in eyes]
            
            if len(eye_centers) == 2:
                center_x = (eye_centers[0][0] + eye_centers[1][0]) // 2
                frame_center = frame.shape[1] // 2
                
                if center_x < frame_center - 50:
                    gaze = "Left"
                elif center_x > frame_center + 50:
                    gaze = "Right"
                else:
                    gaze = "Center"
                
                cv2.putText(frame, f"Gaze: {gaze}", (30, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # FPS 표시
    curr_time = cv2.getTickCount()
    freq = cv2.getTickFrequency()
    fps = freq / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (30, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()