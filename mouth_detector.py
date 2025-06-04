import cv2
import math
import mediapipe as mp
from collections import deque
import numpy as np
import csv
import time


program_start_time = time.time()  # Program start timestamp in seconds

# MediaPipe Face Mesh initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Mouth landmarks indices in MediaPipe Face Mesh
TOP_LIP = 13
BOTTOM_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308
CHIN = 152

# Tracking data structures
face_data = {}         # face_id : tracking info dict
face_lost_counter = {} # face_id : lost frame count

# Parameters
MATCH_THRESHOLD = 50        # pixels, max distance to match faces
LOST_FRAMES_THRESHOLD = 10  # frames to remove lost face


# Utility functions
def euclidean_3d(pt1, pt2):
    """Calculate 3D Euclidean distance between two landmarks."""
    return math.sqrt((pt2.x - pt1.x) ** 2 + (pt2.y - pt1.y) ** 2 + (pt2.z - pt1.z) ** 2)


def calculate_mouth_features(landmarks):
    """Calculate normalized vertical and horizontal mouth openness relative to face height."""
    top = landmarks[TOP_LIP]
    bottom = landmarks[BOTTOM_LIP]
    left = landmarks[LEFT_MOUTH]
    right = landmarks[RIGHT_MOUTH]
    chin = landmarks[CHIN]

    vertical = euclidean_3d(top, bottom)
    horizontal = euclidean_3d(left, right)
    face_height = euclidean_3d(top, chin)

    norm_vertical = vertical / face_height if face_height > 0 else 0
    norm_horizontal = horizontal / face_height if face_height > 0 else 0

    return norm_vertical, norm_horizontal


def update_ema(prev, current, alpha=0.3):
    """Update exponential moving average."""
    if prev is None:
        return current
    return alpha * current + (1 - alpha) * prev


def update_openness_history(history, value, max_len=10):
    """Append value to history deque and return its standard deviation."""
    history.append(value)
    if len(history) > max_len:
        history.popleft()
    return np.std(history)


def get_face_center(landmarks, width, height):
    """Calculate face center pixel coordinates from landmarks."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    center_x = int(np.mean(xs) * width)
    center_y = int(np.mean(ys) * height)
    return center_x, center_y


def match_face(center, face_data):
    """Match detected face center to existing tracked faces based on proximity."""
    for fid, data in face_data.items():
        prev_center = data['center']
        dist = math.sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2)
        if dist < MATCH_THRESHOLD:
            return fid
    return None


def create_new_face(face_data, center):
    """Create new face entry with unique ID."""
    new_id = max(face_data.keys()) + 1 if face_data else 0
    face_data[new_id] = {
        'center': center,
        'openness_ema': None,
        'status': 'Silent',
        'frame_count': 0,
        'openness_history': deque(maxlen=10),
        'warning': False
    }
    return new_id


def log_student_behavior(face_id, status, warning):
    """Log student behavior to CSV with timestamp and warning suppression before 15 minutes."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = time.time() - program_start_time

    # Suppress warnings during first 15 minutes but still log status
    if elapsed_time < 900 and warning:
        warning = False

    with open('student_behavior_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, face_id, status, warning])


def process_mouth(frame):
    """Process frame, detect faces and mouth openness, update status, and visualize."""
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    detected_face_ids = set()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            center = get_face_center(landmarks, width, height)
            matched_id = match_face(center, face_data)
            if matched_id is None:
                matched_id = create_new_face(face_data, center)
            else:
                face_data[matched_id]['center'] = center

            detected_face_ids.add(matched_id)

            norm_vertical, _ = calculate_mouth_features(landmarks)
            prev_ema = face_data[matched_id]['openness_ema']
            smoothed_openness = update_ema(prev_ema, norm_vertical)
            face_data[matched_id]['openness_ema'] = smoothed_openness

            std_dev = update_openness_history(face_data[matched_id]['openness_history'], smoothed_openness)

            # Thresholds for mouth status detection
            TALKING_OPENNESS = 0.12
            MOUTH_OPEN_THRESHOLD = 0.06
            TALKING_STD_THRESHOLD = 0.007

            if smoothed_openness > TALKING_OPENNESS and std_dev > TALKING_STD_THRESHOLD:
                current_status = "Talking"
                warning = True
            elif smoothed_openness > MOUTH_OPEN_THRESHOLD:
                current_status = "Mouth Open"
                warning = False
            else:
                current_status = "Silent"
                warning = False

            # Smooth status changes by requiring consistent frames
            if current_status != face_data[matched_id]['status']:
                face_data[matched_id]['frame_count'] += 1
                if face_data[matched_id]['frame_count'] > 5:
                    face_data[matched_id]['status'] = current_status
                    face_data[matched_id]['warning'] = warning
                    face_data[matched_id]['frame_count'] = 0

                    log_student_behavior(matched_id + 1, current_status, warning)
            else:
                face_data[matched_id]['frame_count'] = 0

            # Visualization (optional)
            color = (0, 255, 0) if face_data[matched_id]['status'] == "Talking" else \
                    (0, 255, 255) if face_data[matched_id]['status'] == "Mouth Open" else (0, 0, 255)

            cv2.putText(frame,
                        f"Student {matched_id + 1}: {face_data[matched_id]['status']}",
                        (center[0] - 60, center[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if face_data[matched_id]['warning']:
                cv2.putText(frame,
                            "WARNING: Talking Detected!",
                            (center[0] - 80, center[1] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Remove lost faces after certain frames of disappearance
    for fid in list(face_data.keys()):
        if fid not in detected_face_ids:
            face_lost_counter[fid] = face_lost_counter.get(fid, 0) + 1
            if face_lost_counter[fid] > LOST_FRAMES_THRESHOLD:
                del face_data[fid]
                del face_lost_counter[fid]
        else:
            face_lost_counter[fid] = 0

    return frame
