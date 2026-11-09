import sys
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import pygame
from gtts import gTTS
import os
import time
import threading
from collections import deque
import matplotlib.pyplot as plt

LINE = "multiwindow"   # "on" = tampil garis, "off" = hilang semua garis, "multiwindow" = 2 window
LINE_COLOR = (255, 224, 0)  # BGR warna biru keemasan
DROWSY_TIME_THRESHOLD = 2.0  # detik

# EAR threshold and consecutive frames
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 45
WARNING_COOLDOWN = 5

pygame.mixer.init()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ============================================================
# CLASS DETECTOR
# ============================================================

class DrowsinessDetector:
    def __init__(self):
        self.ear_history = deque(maxlen=100)
        self.drowsy_frames = 0
        self.last_warning_time = 0
        self.warning_active = False
        self.total_drowsy_count = 0
        self.eye_closed_start = None

    def calculate_ear(self, landmarks, eye_indices):
        eye_points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])

        vertical_1 = distance.euclidean(eye_points[1], eye_points[5])
        vertical_2 = distance.euclidean(eye_points[2], eye_points[4])
        horizontal = distance.euclidean(eye_points[0], eye_points[3])

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def speak_warning(self, text):

        def _play():
            file = f"warning_{int(time.time() * 1000)}.mp3"
            try:
                tts = gTTS(text=text, lang='id')
                tts.save(file)

                pygame.mixer.music.load(file)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                if os.path.exists(file):
                    os.remove(file)

            except Exception as e:
                print("Audio error:", e)

        threading.Thread(target=_play, daemon=True).start()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def draw_landmarks_with_lines(frame, face_landmarks):
    """Draw landmarks with lines (for LINE=on or window_with_line)"""
    mp_drawing.draw_landmarks(
        frame, face_landmarks,
        mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=LINE_COLOR, thickness=1, circle_radius=1
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=LINE_COLOR, thickness=1
        )
    )
    
    mp_drawing.draw_landmarks(
        frame, face_landmarks,
        mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=LINE_COLOR, thickness=1, circle_radius=1
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=LINE_COLOR, thickness=1
        )
    )

def draw_status_text(frame, avg_ear, detector, is_with_line=False):
    """Draw status text on frame with optional window identifier"""
    window_type = "With Lines" if is_with_line else "Without Lines"
    cv2.putText(frame, f"Window: {window_type}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Drowsiness alert if applicable
    if avg_ear < EAR_THRESHOLD and detector.drowsy_frames >= CONSECUTIVE_FRAMES:
        cv2.putText(frame, "DROWSINESS ALERT!", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Status text
    status = "DROWSY" if detector.eye_closed_start else "AWAKE"
    color = (0, 0, 255) if detector.drowsy_frames > 0 else (0, 255, 0)
    
    cv2.putText(frame, f"Status: {status}",
                (frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(frame, f"Drowsy Count: {detector.total_drowsy_count}",
                (frame.shape[1] - 200, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if detector.eye_closed_start:
        closed_time = time.time() - detector.eye_closed_start
        cv2.putText(frame, f"Eyes Closed: {closed_time:.2f}s",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)



# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat diakses! Pastikan kamera terhubung atau tidak sedang dipakai aplikasi lain.")
        sys.exit(1)

    detector = DrowsinessDetector()

    # ================= GRAPH =====================
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.5)
    ax.axhline(y=EAR_THRESHOLD, color='r', linestyle='--', label='Threshold')
    line, = ax.plot([], [], 'b-', label='EAR')
    ax.set_title('Eye Aspect Ratio (EAR) History')
    ax.set_xlabel('Frame')
    ax.set_ylabel('EAR Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show(block=False)

    # ESC detection in matplotlib
    esc_pressed = False

    def on_key(event):
        nonlocal esc_pressed
        if event.key == "escape":
            esc_pressed = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ================= FACE MESH =================
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            if esc_pressed:
                break

            ret, frame = cap.read()
            if not ret:
                print("Camera not accessible!")
                break

            original_frame = cv2.flip(frame, 1)
            
            # Prepare frames for different windows
            if LINE == "multiwindow":
                frame_with_line = original_frame.copy()
                frame_without_line = original_frame.copy()
            elif LINE == "on":
                frame_with_line = original_frame.copy()
                frame_without_line = None
            else:  # LINE == "off"
                frame_with_line = None
                frame_without_line = original_frame.copy()

            rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark

                # ================= EAR CALC ============================
                left_ear = detector.calculate_ear(landmarks, LEFT_EYE)
                right_ear = detector.calculate_ear(landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2

                detector.ear_history.append(avg_ear)

                # GRAPH UPDATE
                line.set_data(range(len(detector.ear_history)),
                              list(detector.ear_history))
                fig.canvas.draw()
                fig.canvas.flush_events()

                # ================= DRAW LANDMARKS BASED ON MODE ==============
                if LINE == "multiwindow":
                    # Window 1: With lines
                    draw_landmarks_with_lines(frame_with_line, face_landmarks)
                    draw_status_text(frame_with_line, avg_ear, detector, is_with_line=True)
                    
                    # Window 2: Without lines (just show face)
                    draw_status_text(frame_without_line, avg_ear, detector, is_with_line=False)
                    
                elif LINE == "on":
                    # Single window with lines
                    draw_landmarks_with_lines(frame_with_line, face_landmarks)
                    draw_status_text(frame_with_line, avg_ear, detector, is_with_line=True)
                    
                else:  # LINE == "off"
                    # Single window without lines
                    draw_status_text(frame_without_line, avg_ear, detector, is_with_line=False)

                # ================= DROWSINESS CHECK =====================
                current_time = time.time()

                if avg_ear < EAR_THRESHOLD:
                    if detector.eye_closed_start is None:
                        detector.eye_closed_start = current_time

                    closed_duration = current_time - detector.eye_closed_start

                    if closed_duration >= DROWSY_TIME_THRESHOLD:
                        if current_time - detector.last_warning_time > WARNING_COOLDOWN:
                            detector.total_drowsy_count += 1
                            detector.speak_warning(
                                "Peringatan! Anda mengantuk. Segera Istirahat!"
                            )
                            detector.last_warning_time = current_time
                else:
                    detector.eye_closed_start = None

            # ================= DISPLAY WINDOWS =====================
            if LINE == "multiwindow":
                cv2.imshow("Drowsiness Detection - With Face Mesh", frame_with_line)
                cv2.imshow("Drowsiness Detection - Clean View", frame_without_line)
                
                # Adjust window positions
                cv2.moveWindow("Drowsiness Detection - With Face Mesh", 50, 100)
                cv2.moveWindow("Drowsiness Detection - Clean View", 
                              frame_with_line.shape[1] + 100, 100)
                
            elif LINE == "on":
                cv2.imshow("Drowsiness Detection", frame_with_line)
                cv2.moveWindow("Drowsiness Detection", 50, 100)
                
            else:  # LINE == "off"
                cv2.imshow("Drowsiness Detection", frame_without_line)
                cv2.moveWindow("Drowsiness Detection", 50, 100)

            # ESC from camera window
            key = cv2.waitKey(1)
            if key == 27:
                break

    cap.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
    # Wait a bit to ensure windows are closed
    time.sleep(0.1)
    
    # Close matplotlib
    plt.close()
    pygame.mixer.quit()
    
    print(f"\nProgram terminated. Total drowsy events detected: {detector.total_drowsy_count}")


if __name__ == "__main__":
    main()