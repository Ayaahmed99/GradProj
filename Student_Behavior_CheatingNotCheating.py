import cv2
import time
import csv
from datetime import datetime
import numpy as np
from ultralytics import YOLO
from face_rec import FaceRecognition
import psycopg2

# Database config
db_config = {
    "host": "localhost",
    "dbname": "face_recognition",
    "user": "postgres",
    "password": "root"
}

class StudentBehaviorMonitor:
    def __init__(self, model_path, log_path="behavior_log.csv"):
        self.model = YOLO(model_path)
        print("Loaded classes:", self.model.names)

        self.face_recognizer = FaceRecognition()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Webcam not accessible.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.prev_time = time.time()

        self.cheating_counts = {}    # {student_name: count}
        self.warning_counts = {}     # {student_name: count}
        self.last_warning_times = {} # {student_name: last_warning_timestamp}
        self.warning_threshold = 3
        self.warning_cooldown = 10  # seconds

        self.log_path = log_path
        self.csv_file = open(self.log_path, mode="w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["Timestamp", "Student ID", "Student Name", "Detected Class", "Confidence",
                              "Behavior", "Action", "Warning Count"])

        self.conf_threshold = 0.75

        # DB connection
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()

        self.course_name = "Physics"  # Hardcoded or dynamically assigned

    def log_to_database(self, student_id, student_name, behaviour_type, warning_count, frame):
        try:
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()

            query = """
            INSERT INTO behaviour (student_id, student_name, course_name, behaviour_type, warnings, captured_frame)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.cursor.execute(query, (
                student_id,
                student_name,
                self.course_name,
                behaviour_type,
                warning_count,
                psycopg2.Binary(img_bytes)
            ))
            self.conn.commit()
        except Exception as e:
            print(f"[DB Insert Error]: {e}")

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (800, 800))
        results = self.model(frame_resized, stream=True)
        face_names = self.face_recognizer.get_names_for_faces(frame_resized)  # [(id, name, (x1,y1,x2,y2))]

        current_time = time.time()

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                class_name = self.model.names[cls_id].lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                student_id = "Unknown"
                student_name = "Unknown"
                min_dist = float('inf')
                for sid, name, (fx1, fy1, fx2, fy2) in face_names:
                    face_center = ((fx1 + fx2) // 2, (fy1 + fy2) // 2)
                    dist = np.linalg.norm(np.array(box_center) - np.array(face_center))
                    if dist < min_dist:
                        min_dist = dist
                        student_id = sid
                        student_name = name

                if student_name not in self.cheating_counts:
                    self.cheating_counts[student_name] = 0
                    self.warning_counts[student_name] = 0
                    self.last_warning_times[student_name] = 0

                detected_behavior = "Not Cheating"
                action = "None"
                label_color = (0, 255, 0)

                if class_name == "cheating" and conf >= self.conf_threshold:
                    self.cheating_counts[student_name] += 1
                    detected_behavior = "Cheating"

                    if self.cheating_counts[student_name] >= self.warning_threshold:
                        if current_time - self.last_warning_times[student_name] >= self.warning_cooldown:
                            self.warning_counts[student_name] += 1
                            self.last_warning_times[student_name] = current_time
                            action = "Warning"
                            label_color = (0, 0, 255)

                            # ✅ Only log if student_id and name are not Unknown
                            if student_id != "Unknown" and student_name != "Unknown":
                                self.log_to_database(student_id, student_name, detected_behavior,
                                                     self.warning_counts[student_name], frame_resized)

                        else:
                            action = "Already Warned"
                            label_color = (0, 165, 255)
                    else:
                        action = "Monitor"
                        label_color = (0, 165, 255)
                else:
                    detected_behavior = "Not Cheating"
                    label_color = (255, 0, 0)

                # Draw bounding box and labels
                label = f"{student_name}: {class_name} ({conf:.2f})"
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), label_color, 2)
                cv2.putText(frame_resized, label, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
                cv2.putText(frame_resized, f"Behavior: {detected_behavior}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
                cv2.putText(frame_resized, f"Warnings: {self.warning_counts[student_name]}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

                # Log to CSV
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.writer.writerow([
                    timestamp,
                    student_id,
                    student_name,
                    class_name,
                    f"{conf:.2f}",
                    detected_behavior,
                    action,
                    self.warning_counts[student_name]
                ])
                self.csv_file.flush()

        return frame_resized

    def detect_and_log_behavior(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to grab frame.")
                    break

                processed_frame = self.process_frame(frame)

                current_time = time.time()
                fps = 1 / (current_time - self.prev_time)
                self.prev_time = current_time

                cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Student Behavior Monitor", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Exception occurred: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.csv_file.close()
        self.cursor.close()
        self.conn.close()
        print(f"Session ended. Log saved to '{self.log_path}'")
        for student, warnings in self.warning_counts.items():
            cheating = self.cheating_counts.get(student, 0)
            print(f"{student}: {warnings} warnings, {cheating} cheating detections")


if __name__ == "__main__":
    MODEL_PATH = r"best1.pt"
    monitor = StudentBehaviorMonitor(MODEL_PATH)
    monitor.detect_and_log_behavior()
