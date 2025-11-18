import cv2
import mediapipe as mp
import pygame
import numpy as np
import os

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- Initialize pygame for sound ---
pygame.mixer.init()

# Load piano sounds 
notes = ['A.mp3','B.mp3','C.mp3', 'D.mp3', 'E.mp3', 'F.mp3', 'G.mp3']
sound_dir = "sounds"
sounds = []

for n in notes:
    path = os.path.join(sound_dir, n)
    if os.path.exists(path):
        sounds.append(pygame.mixer.Sound(path))
    else:
        print(f"Missing sound file: {path}")
        sounds.append(None)

# --- Webcam setup ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Visual piano key layout
key_colors = [(255, 255, 255)] * len(notes)
pressed = [False] * len(notes)
finger_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

print("Five-Finger Virtual Piano Ready! Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    key_width = w // len(notes)

    # Draw piano keys
    for i in range(len(notes)):
        x1, x2 = i * key_width, (i + 1) * key_width
        color = (0, 200, 255) if pressed[i] else key_colors[i]
        cv2.rectangle(frame, (x1, h - 200), (x2, h), color, -1)
        cv2.rectangle(frame, (x1, h - 200), (x2, h), (0, 0, 0), 2)
        cv2.putText(frame, notes[i][0], (x1 + 40, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    # Process hand landmarks
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Reset key press states
    current_pressed = [False] * len(notes)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for i, fid in enumerate(finger_ids):
                lm = hand_landmarks.landmark[fid]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 12, (0, 0, 255), -1)

                if h - 200 < y < h:
                    key_index = min(x // key_width, len(notes) - 1)
                    current_pressed[key_index] = True

                    # Play sound only if newly pressed
                    if not pressed[key_index]:
                        pressed[key_index] = True
                        if sounds[key_index]:
                            sounds[key_index].play()

    # Update pressed states (release notes if finger lifted)
    for i in range(len(notes)):
        if not current_pressed[i]:
            pressed[i] = False

    cv2.imshow("Five-Finger Piano", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
