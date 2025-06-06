import cv2
import mediapipe as mp
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Variables
canvas = None
prev_x, prev_y = 0, 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get coordinates of index and middle fingers
        index_finger_tip = hand_landmarks.landmark[8]
        middle_finger_tip = hand_landmarks.landmark[12]
        index_finger_bottom = hand_landmarks.landmark[6]
        middle_finger_bottom = hand_landmarks.landmark[10]

        x = int(index_finger_tip.x * w)
        y = int(index_finger_tip.y * h)

        # Detect finger states (up or down)
        index_up = index_finger_tip.y < index_finger_bottom.y
        middle_up = middle_finger_tip.y < middle_finger_bottom.y

        # Drawing mode (only index finger up)
        if index_up and not middle_up:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)
            prev_x, prev_y = x, y
            cv2.circle(frame, (x, y), 8, (255, 0, 0), cv2.FILLED)

        # Eraser mode (both fingers up)
        elif index_up and middle_up:
            cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)
            prev_x, prev_y = 0, 0
            cv2.circle(frame, (x, y), 30, (0, 0, 255), 2)

        # Reset previous positions when fingers are not up
        else:
            prev_x, prev_y = 0, 0

    # Combine drawing with camera feed
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
    drawing_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(frame_bg, drawing_fg)

    cv2.imshow("Air Drawing - Press 'q' to Quit", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
