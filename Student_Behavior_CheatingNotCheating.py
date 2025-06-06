import cv2
import time
import csv
from datetime import datetime
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

        self.student_id = 1
        self.cheating_count = 0
        self.warning_count = 0
        self.warning_threshold = 3

        # Cooldown mechanism
        self.last_warning_time = 0
        self.warning_cooldown = 10  # seconds

        # Logging setup
        self.log_path = log_path
        self.csv_file = open(self.log_path, mode="w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["Timestamp", "Student ID", "Detected Class", "Confidence",
                              "Behavior", "Action", "Warning Count"])

        # Confidence threshold for cheating
        self.conf_threshold = 0.75

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (800, 800))
        results = self.model(frame_resized, stream=True)

        current_time = time.time()
        detected_behavior = "Not Cheating"
        action = "None"
        label_color = (0, 255, 0)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                class_name = self.model.names[cls_id].lower()

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if class_name == "cheating" and conf >= self.conf_threshold:
                    self.cheating_count += 1
                    detected_behavior = "Cheating"

                    # Warning logic with cooldown
                    if self.cheating_count >= self.warning_threshold:
                        if current_time - self.last_warning_time >= self.warning_cooldown:
                            self.warning_count += 1
                            self.last_warning_time = current_time
                            action = "Warning"
                            label_color = (0, 0, 255)
                        else:
                            action = "Already Warned"
                            label_color = (0, 165, 255)
                    else:
                        action = "Monitor"
                        label_color = (0, 165, 255)

                # Draw rectangle and labels
                label = f"Student {self.student_id}: {class_name} ({conf:.2f})"
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), label_color, 2)
                cv2.putText(frame_resized, label, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
                cv2.putText(frame_resized, f"Behavior: {detected_behavior}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
                cv2.putText(frame_resized, f"Warnings: {self.warning_count}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

                # Log behavior to CSV
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.writer.writerow([
                    timestamp,
                    f"Student {self.student_id}",
                    class_name,
                    f"{conf:.2f}",
                    detected_behavior,
                    action,
                    self.warning_count
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

                # FPS calculation
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
        print(f"Student {self.student_id}: {self.warning_count} warnings, {self.cheating_count} cheating detections")


if __name__ == "__main__":
    MODEL_PATH = r"D:\GradProj\MAINFILES\Models\LastModel.pt"
    monitor = StudentBehaviorMonitor(MODEL_PATH)
    monitor.detect_and_log_behavior()
