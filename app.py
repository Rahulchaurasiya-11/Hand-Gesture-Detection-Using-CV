from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math
import time  # noqa: F401

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def is_fist(lm):
    wrist = (lm[0].x, lm[0].y)
    tips = [lm[i] for i in (4, 8, 12, 16, 20)]
    dists = [distance((t.x, t.y), wrist) for t in tips]
    return max(dists) < 0.08


def is_thumb_up(lm):
    wrist = (lm[0].x, lm[0].y)
    thumb_tip = (lm[4].x, lm[4].y)
    index_mcp = (lm[5].x, lm[5].y)
    return thumb_tip[1] < wrist[1] and distance(thumb_tip, index_mcp) > 0.06


def is_palm_open(lm):
    wrist = (lm[0].x, lm[0].y)
    tips = [lm[i] for i in (4, 8, 12, 16, 20)]
    dists = [distance((t.x, t.y), wrist) for t in tips]
    return min(dists) > 0.12


def is_victory(lm):
    wrist = (lm[0].x, lm[0].y)
    index_tip = (lm[8].x, lm[8].y)
    middle_tip = (lm[12].x, lm[12].y)
    ring_tip = (lm[16].x, lm[16].y)
    pinky_tip = (lm[20].x, lm[20].y)
    cond1 = distance(index_tip, wrist) > 0.12 and distance(middle_tip, wrist) > 0.12
    cond2 = distance(ring_tip, wrist) < 0.09 and distance(pinky_tip, wrist) < 0.09
    return cond1 and cond2


def is_ok(lm):
    thumb_tip = (lm[4].x, lm[4].y)
    index_tip = (lm[8].x, lm[8].y)
    return distance(thumb_tip, index_tip) < 0.05


def count_fingers(lm):
    fingers = []
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)
    fingers.append(1 if lm[4].x < lm[3].x else 0)
    return sum(fingers)


def generate_frames():
    cap = cv2.VideoCapture(0)
    flash_alpha = 0
    flash_color = (0, 255, 255)

    with mp_hands.Hands(min_detection_confidence=0.6,
                        min_tracking_confidence=0.5, max_num_hands=2) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            label = "No hand"
            flash_triggered = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm = hand_landmarks.landmark
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if is_fist(lm):
                        label = "✊ Fist"
                        flash_triggered = True
                    elif is_thumb_up(lm):
                        label = "👍 Thumbs Up"
                        flash_triggered = True
                    elif is_victory(lm):
                        label = "✌️ Victory"
                        flash_triggered = True
                    elif is_palm_open(lm):
                        label = "🖐️ Open Palm"
                        flash_triggered = True
                    elif is_ok(lm):
                        label = "👌 OK Sign"
                        flash_triggered = True
                    else:
                        count = count_fingers(lm)
                        if 1 <= count <= 5:
                            label = f"Number: {count}"
                            flash_triggered = True

                    wrist = (int(lm[0].x * w), int(lm[0].y * h))
                    cv2.circle(frame, wrist, 20, (0, 255, 255), 3)
                    cv2.putText(frame, "Movement", (wrist[0] + 25, wrist[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, 'Team: RMR Team', (w//2 - 150, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, f'Gesture: {label}', (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if flash_triggered:
                flash_alpha = 150

            if flash_alpha > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), flash_color, -1)
                cv2.addWeighted(overlay, flash_alpha / 255, frame, 1 - flash_alpha / 255, 0, frame)
                flash_alpha -= 10

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
