import cv2
import mediapipe as mp   
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def is_thumb_up(landmarks):
    # thumb tip (4) above wrist (0) in y (screen coords: smaller y = up)
    # and thumb folded direction check: tip far from index MCP
    wrist = (landmarks[0].x, landmarks[0].y)
    thumb_tip = (landmarks[4].x, landmarks[4].y)
    index_mcp = (landmarks[5].x, landmarks[5].y)
    return thumb_tip[1] < wrist[1] and distance(thumb_tip, index_mcp) > 0.06

def is_fist(landmarks):
    # All finger tips close to wrist
    wrist = (landmarks[0].x, landmarks[0].y)
    tips = [landmarks[i] for i in (4, 8, 12, 16, 20)]
    dists = [distance((t.x, t.y), wrist) for t in tips]
    return max(dists) < 0.08

def is_palm_open(landmarks):
    # All finger tips far from wrist and relatively above wrist
    wrist = (landmarks[0].x, landmarks[0].y)
    tips = [landmarks[i] for i in (4, 8, 12, 16, 20)]
    dists = [distance((t.x, t.y), wrist) for t in tips]
    return min(dists) > 0.12

def is_victory(landmarks):
    # Index and middle fingers extended, others folded
    wrist = (landmarks[0].x, landmarks[0].y)
    index_tip = (landmarks[8].x, landmarks[8].y)
    middle_tip = (landmarks[12].x, landmarks[12].y)
    ring_tip = (landmarks[16].x, landmarks[16].y)
    pinky_tip = (landmarks[20].x, landmarks[20].y)
    thumb_tip = (landmarks[4].x, landmarks[4].y)
    # index & middle far from wrist, ring/pinky/thumb close
    cond1 = distance(index_tip, wrist) > 0.12 and distance(middle_tip, wrist) > 0.12
    cond2 = distance(ring_tip, wrist) < 0.09 and distance(pinky_tip, wrist) < 0.09
    return cond1 and cond2

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        label = "No hand"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark
                # call heuristic checks
                if is_fist(lm):
                    label = "Fist"
                elif is_thumb_up(lm):
                    label = "OK"
                elif is_victory(lm):
                    label = "Victory (V)"
                elif is_palm_open(lm):
                    label = "Open Palm"
                else:
                    label = "Unknown"

                # Show coordinates of wrist for debugging (optional)
                wrist = lm[0]
                h, w, _ = frame.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
        
        cv2.putText(frame, f'Gesture: {label}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Heuristic Hand Gesture', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
