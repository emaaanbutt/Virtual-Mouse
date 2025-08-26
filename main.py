import cv2
import mediapipe as mp
import pyautogui
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

smooth_x, smooth_y = None, None
alpha = 0.25 

prev_wy = None 

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, c = frame.shape

                index_finger = hand_landmarks.landmark[8]
                thumb = hand_landmarks.landmark[4]
                wrist = hand_landmarks.landmark[0]

                ix, iy = int(index_finger.x * w), int(index_finger.y * h)
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                wx, wy = int(wrist.x * w), int(wrist.y * h)

                cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
                cv2.circle(frame, (tx, ty), 10, (0, 0, 255), -1)
                cv2.circle(frame, (wx, wy), 10, (255, 0, 0), -1)

                screen_x = int(index_finger.x * screen_w)
                screen_y = int(index_finger.y * screen_h)

                if smooth_x is None:
                    smooth_x, smooth_y = screen_x, screen_y
                else:
                    smooth_x = int(alpha*screen_x + (1-alpha)*smooth_x)
                    smooth_y = int(alpha*screen_y + (1-alpha)*smooth_y)

                pyautogui.moveTo(smooth_x, smooth_y)

                distance = math.hypot(tx - ix, ty - iy)
                if distance < 40:
                    pyautogui.click()
                    pyautogui.sleep(0.2)

                if prev_wy is not None:
                    dy = wy - prev_wy  
                    if abs(dy) > 10:  
                        pyautogui.scroll(-int(dy * 2)) 
                prev_wy = wy

        else:
            prev_wy = None  

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
