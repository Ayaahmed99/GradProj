import cv2
import numpy as np
from collections import deque
import math
import mediapipe as mp
import psycopg2
from datetime import datetime
import time

#from main import YOLOv8FaceDetector  # Import the YOLOv8 face detector
from main import YOLOv8FaceDetector
from face_rec import FaceRecognition  # Import your face recognition class

# Database config (replace with your actual credentials)
db_config = {
    "host": "localhost",
    "dbname": "face_recognition",
    "user": "postgres",
    "password": "root"
}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

TOP_LIP, BOTTOM_LIP = 13, 14
LEFT_MOUTH, RIGHT_MOUTH = 78, 308
CHIN = 152

face_data = {}
face_lost_counter = {}

MATCH_THRESHOLD = 50
LOST_FRAMES_THRESHOLD = 10

face_recognition = FaceRecognition()  # Initialize face recognition class

# Track warnings per student id + course
warning_counts = {}

course_name = "Chemistry"  # or dynamic

def euclidean_3d(pt1, pt2):
    return math.sqrt((pt2.x - pt1.x) ** 2 + (pt2.y - pt1.y) ** 2 + (pt2.z - pt2.z) ** 2)

def calculate_mouth_features(landmarks):
    vertical = euclidean_3d(landmarks[TOP_LIP], landmarks[BOTTOM_LIP])
    horizontal = euclidean_3d(landmarks[LEFT_MOUTH], landmarks[RIGHT_MOUTH])
    face_height = euclidean_3d(landmarks[TOP_LIP], landmarks[CHIN])
    return vertical / face_height if face_height > 0 else 0, horizontal / face_height if face_height > 0 else 0

def update_ema(prev, current, alpha=0.3):
    return current if prev is None else alpha * current + (1 - alpha) * prev

def update_openness_history(history, value, max_len=10):
    history.append(value)
    if len(history) > max_len:
        history.popleft()
    return np.std(history)

def get_face_center(landmarks, width, height):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return int(np.mean(xs) * width), int(np.mean(ys) * height)

def match_face(center, face_data):
    for fid, data in face_data.items():
        prev_center = data['center']
        if math.sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2) < MATCH_THRESHOLD:
            return fid
    return None

def create_new_face(face_data, center):
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

def log_student_behavior_to_db(student_id, student_name, course_name, status, warning_count, frame):
    if student_id == "Unknown" or student_name == "Unknown":
        return
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Check if entry exists for student_id + course_name
        cursor.execute("""
            SELECT warnings FROM behaviour
            WHERE student_id = %s AND course_name = %s
            ORDER BY created_at DESC LIMIT 1
        """, (student_id, course_name))
        row = cursor.fetchone()

        # Encode image frame
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        created_at = datetime.now()

        if row:
            # Existing record found, increment warning count
            new_warning_count = row[0] + 1
            cursor.execute("""
                INSERT INTO behaviour (student_id, student_name, course_name, behaviour_type, warnings, captured_frame, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (student_id, student_name, course_name, status, new_warning_count, psycopg2.Binary(image_bytes), created_at))
        else:
            # No record found, insert new with warning_count=1
            cursor.execute("""
                INSERT INTO behaviour (student_id, student_name, course_name, behaviour_type, warnings, captured_frame, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (student_id, student_name, course_name, status, 1, psycopg2.Binary(image_bytes), created_at))

        conn.commit()
        cursor.close()
        conn.close()

        # Update warning_counts dict for session tracking
        warning_counts[(student_id, course_name)] = warning_counts.get((student_id, course_name), 0) + 1

    except Exception as e:
        print(f"DB Insert Error: {e}")

def process_mouth(frame, box, recognized_faces):  # Now also accepts the bounding box
    x1, y1, x2, y2 = box
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    detected_face_ids = set()
    student_name = "Unknown"
    student_id = "Unknown"

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        center = get_face_center(landmarks, width, height)

        matched_id = match_face(center, face_data)
        if matched_id is None:
            matched_id = create_new_face(face_data, center)
        else:
            face_data[matched_id]['center'] = center
        detected_face_ids.add(matched_id)

        # Find recognized student
        for rid, rname, bbox in recognized_faces:
                # The loop and box information is useless
            student_id = rid
            student_name = rname
            break

        norm_vertical, _ = calculate_mouth_features(landmarks)
        smoothed_openness = update_ema(face_data[matched_id]['openness_ema'], norm_vertical)
        face_data[matched_id]['openness_ema'] = smoothed_openness

        std_dev = update_openness_history(face_data[matched_id]['openness_history'], smoothed_openness)

        TALKING_OPENNESS = 0.12
        MOUTH_OPEN_THRESHOLD = 0.06
        TALKING_STD_THRESHOLD = 0.007

        if smoothed_openness > TALKING_OPENNESS and std_dev > TALKING_STD_THRESHOLD:
            current_status, warning = "Talking", True
        elif smoothed_openness > MOUTH_OPEN_THRESHOLD:
            current_status, warning = "Mouth Open", False
        else:
            current_status, warning = "Silent", False

        if current_status != face_data[matched_id]['status']:
            face_data[matched_id]['frame_count'] += 1
            if face_data[matched_id]['frame_count'] > 5:
                face_data[matched_id]['status'] = current_status
                face_data[matched_id]['warning'] = warning
                face_data[matched_id]['frame_count'] = 0

                if warning and student_id != "Unknown" and student_name != "Unknown":
                    key = (student_id, course_name)
                    warning_count = warning_counts.get(key, 0) + 1
                    log_student_behavior_to_db(student_id, student_name, course_name, current_status, warning_count, frame)
                    warning_counts[key] = warning_count
        else:
            face_data[matched_id]['frame_count'] = 0

        color = (0, 255, 0) if current_status == "Talking" else (0, 255, 255) if current_status == "Mouth Open" else (0, 0, 255)

        # Draw rectangle and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{student_name}: {current_status}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if warning:
            cv2.putText(frame, "WARNING: Talking Detected!", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # Cleanup lost faces
    for fid in list(face_data.keys()):
        if fid not in detected_face_ids:
            face_lost_counter[fid] = face_lost_counter.get(fid, 0) + 1
            if face_lost_counter[fid] > LOST_FRAMES_THRESHOLD:
                del face_data[fid]
                del face_lost_counter[fid]
        else:
            face_lost_counter[fid] = 0

    return frame


if __name__ == '__main__':
    model_path = "yolov8n-face.onnx"
    conf_threshold = 0.45
    iou_threshold = 0.5

    face_detector = YOLOv8FaceDetector(model_path, conf_threshold, iou_threshold)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get face detections from YOLOv8
        face_boxes = face_detector.get_face_boxes(frame)

        # Process each detected face
        for box in face_boxes:
            x1, y1, x2, y2 = box
            face_img = frame[y1:y2, x1:x2]  # Crop the face
            recognized_faces = []
            if face_img.size > 0:
                name_data = face_recognition.get_names_for_faces(face_img)  # face_recognition on cropped face
                for rid, rname, _ in name_data:
                    recognized_faces.append((rid, rname, box))  #Keep original box

            frame = process_mouth(frame, box, recognized_faces)  # Pass the original frame, the box and the identified face

        cv2.imshow("YOLOv8 Face Detection - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()