import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time
import winsound

# ================= CONFIG =================
frame_reduction = 90
click_threshold = 28
alpha_smooth = 0.3
scroll_sensitivity = 20
# ==========================================

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

# ================= KALMAN =================
kalman = cv2.KalmanFilter(4, 2)

kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)

kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.02

kalman.statePre = np.zeros((4,1), np.float32)
kalman.statePost = np.zeros((4,1), np.float32)

# ================= ESTADO =================
prev_x, prev_y = 0, 0
clicking = False

def beep():
    winsound.Beep(800, 80)

def fingers_up(lm):
    return {
        "index": lm[8].y < lm[6].y,
        "middle": lm[12].y < lm[10].y,
        "ring": lm[16].y < lm[14].y,
        "pinky": lm[20].y < lm[18].y
    }

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    cv2.rectangle(frame,
                  (frame_reduction, frame_reduction),
                  (w - frame_reduction, h - frame_reduction),
                  (0, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            lm = hand_landmarks.landmark
            fingers = fingers_up(lm)

            # ================= INDEX =================
            x = int(lm[8].x * w)
            y = int(lm[8].y * h)

            x = max(frame_reduction, min(x, w - frame_reduction))
            y = max(frame_reduction, min(y, h - frame_reduction))

            sx = (x - frame_reduction) * screen_w / (w - 2 * frame_reduction)
            sy = (y - frame_reduction) * screen_h / (h - 2 * frame_reduction)

            # ================= KALMAN =================
            kalman.predict()

            measurement = np.array([[np.float32(sx)],
                                    [np.float32(sy)]])

            kalman.correct(measurement)

            kx = kalman.statePost[0][0]
            ky = kalman.statePost[1][0]

            final_x = prev_x + (kx - prev_x) * alpha_smooth
            final_y = prev_y + (ky - prev_y) * alpha_smooth

            # ================= SCROLL (2 dedos) =================
            if fingers["index"] and fingers["middle"] and not fingers["ring"]:
                pyautogui.scroll(-int((lm[8].y - lm[12].y) * scroll_sensitivity * 100))
                cv2.putText(frame, "SCROLL MODE", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            else:
                # ================= CURSOR =================
                pyautogui.moveTo(final_x, final_y)
                prev_x, prev_y = final_x, final_y

                # ================= CLICK =================
                x2 = int(lm[4].x * w)
                y2 = int(lm[4].y * h)

                distancia = math.hypot(x2 - x, y2 - y)

                if distancia < click_threshold:
                    if not clicking:
                        pyautogui.click()
                        beep()
                        clicking = True

                        cv2.putText(frame, "CLICK", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    clicking = False

            # ================= VISUAL =================
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (255, 0, 0), -1)
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 255), 2)

    cv2.imshow("IRON MAN CONTROL SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()