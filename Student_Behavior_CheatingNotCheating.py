import cv2
import time
import csv
from datetime import datetime
from collections import deque
from ultralytics import YOLO


class StudentBehaviorMonitor:
    def __init__(self, model_path, log_path="behavior_log.csv"):
        self.model = YOLO(model_path)
        print("Loaded classes:", self.model.names)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Webcam not accessible.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.prev_time = time.time()

        self.cheating_counts = {}  # Track number of cheating detections per student
        self.warning_counts = {}   # Track number of warnings issued per student
        self.warning_threshold = 3  # Only warn at 3rd detection

        self.log_path = log_path
        self.csv_file = open(self.log_path, mode="w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["Timestamp", "Student ID", "Detected Class", "Confidence",
                              "Behavior", "Action", "Warning Count"])

        self.conf_threshold = 0.6

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (800, 800))
        results = self.model(frame_resized, stream=True)

        student_id = 1
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                class_name = self.model.names[cls_id].lower()

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_color = (0, 255, 0)
                behavior_text = "Not Cheating"
                action = "None"

                # Initialize counters if first time
                if student_id not in self.cheating_counts:
                    self.cheating_counts[student_id] = 0
                    self.warning_counts[student_id] = 0

                if class_name == "cheating" and conf >= self.conf_threshold:
                    self.cheating_counts[student_id] += 1

                    if self.cheating_counts[student_id] == self.warning_threshold:
                        behavior_text = "Cheating"
                        action = "Warning"
                        label_color = (0, 0, 255)
                        self.warning_counts[student_id] += 1
                    else:
                        behavior_text = "Cheating"
                        action = "None"
                        label_color = (0, 165, 255)  # Orange for early detections
                else:
                    behavior_text = "Not Cheating"
                    label_color = (0, 255, 0)

                # Draw on frame
                label = f"Student {student_id}: {class_name} ({conf:.2f})"
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), label_color, 2)
                cv2.putText(frame_resized, label, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
                cv2.putText(frame_resized, f"Behavior: {behavior_text}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

                # Log data
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.writer.writerow([
                    timestamp,
                    f"Student {student_id}",
                    class_name,
                    f"{conf:.2f}",
                    behavior_text,
                    action,
                    self.warning_counts[student_id]
                ])
                self.csv_file.flush()

                student_id += 1

        return frame_resized

    def detect_and_log_behavior(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to grab frame.")
                    break

                processed_frame = self.process_frame(frame)

                # Display FPS
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
        print(f"Session ended. Log saved to '{self.log_path}'")


if __name__ == "__main__":
    MODEL_PATH = r"D:\GradProj\MAINFILES\Models\LastModel.pt"
    monitor = StudentBehaviorMonitor(MODEL_PATH)
    monitor.detect_and_log_behavior()
