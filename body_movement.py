import cv2
import time
import mediapipe as mp
from ultralytics import YOLO



class BodyMovementDetector:
    def __init__(self, model_path="yolo8n.pt"):  # Added model path as parameter
        # Initialize Mediapipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Load YOLOv8 Pose model
        self.model = YOLO(model_path)
        print(self.model.names)  # Prints the class names recognized by your model

        # Initialize FPS calculation
        self.prev_time = time.time()

    def process_body_movement(self, frame): #Changed to take frame as argument
        # Setup Mediapipe Pose model
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # Resize frame for better performance
            frame = cv2.resize(frame, (640, 480))
            frame_yolo = frame.copy()

            # Process frame with YOLO Pose estimation
            results_yolo = self.model(frame_yolo, stream=False) #stream=False to ensure complete result

            annotated_frame = frame.copy()  # Start with a clean frame

            # Annotate YOLO results
            for result in results_yolo:
                annotated_frame = result.plot()

            # Convert frame to RGB for Mediapipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False  # Improves performance

            # Process frame for Mediapipe Pose detection
            results_mp = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True  # Allow drawing

            # Draw pose landmarks if detected
            if results_mp.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results_mp.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                )

            # FPS Calculation
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time

            # Display FPS on frame
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated_frame #Return processed frame