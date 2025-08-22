import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

smooth_x, smooth_y = None, None
alpha = 0.25 

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

                # Index finger landmarks
                index_finger = hand_landmarks.landmark[8]

                # convert normalized landmark coords to camera pixels
                ix, iy = int(index_finger.x * w), int(index_finger.y * h)

                cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)

                 # convert normalized landmark coords to screen pixels
                screen_x = int(index_finger.x * screen_w)
                screen_y = int(index_finger.y * screen_h)

                # smoothing (optional)
                if smooth_x is None:
                    smooth_x, smooth_y = screen_x, screen_y
                else:
                    smooth_x = int(alpha*screen_x + (1-alpha)*smooth_x)
                    smooth_y = int(alpha*screen_y + (1-alpha)*smooth_y)

                # move cursor
                pyautogui.moveTo(smooth_x, smooth_y)
               

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
