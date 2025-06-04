import cv2
import numpy as np
import os
import dlib
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognitionMovement:
    def __init__(self, dataset_path="known_faces"):
        # Initialize InsightFace (Face Recognition)
        self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(320, 320))

        # Load dataset and embeddings
        self.dataset_path = dataset_path
        self.class_names, self.embeddings = self.load_dataset()
        print(f"Dataset Loaded. Found {len(self.class_names)} face(s).")

        # Initialize dlib face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            r"D:\GradProj\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
        )

    def load_dataset(self):
        """Load known face images and compute normalized embeddings."""
        names = []
        embeddings = []

        for file_name in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, file_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Skipping {file_name} (Invalid image)")
                continue

            faces = self.app.get(img)
            if faces:
                embedding = faces[0].embedding
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
                embeddings.append(embedding)
                student_id_name = os.path.splitext(file_name)[0]  # e.g. "12345_JohnDoe"
                names.append(student_id_name)
            else:
                print(f"No face detected in {file_name}")

        return names, np.array(embeddings)

    def recognize_face(self, frame):
        """Recognize faces in a frame using cosine similarity."""
        faces = self.app.get(frame)
        labeled_frame = frame.copy()
        labels = []

        if not faces:
            return labeled_frame, labels

        for face in faces:
            best_match = "Unknown"
            face_embedding = face.embedding / np.linalg.norm(face.embedding)

            if len(self.embeddings) > 0:
                similarities = cosine_similarity(self.embeddings, face_embedding.reshape(1, -1)).flatten()
                max_index = np.argmax(similarities)
                max_similarity = similarities[max_index]

                if max_similarity > 0.3:
                    best_match = self.class_names[max_index]

            labels.append(best_match)

            # Use bbox as [x1, y1, x2, y2]
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(labeled_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(labeled_frame, best_match, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return labeled_frame, labels

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180.0 else 360 - angle

    def detect_face_movement(self, shape):
        """Detect head direction based on landmarks."""
        left_eye = (shape.part(36).x, shape.part(36).y)
        right_eye = (shape.part(45).x, shape.part(45).y)
        nose_tip = (shape.part(30).x, shape.part(30).y)
        jaw = (shape.part(8).x, shape.part(8).y)

        angle_left = self.calculate_angle(left_eye, nose_tip, jaw)
        angle_right = self.calculate_angle(right_eye, nose_tip, jaw)

        # Threshold-based simple direction classification
        if angle_left > angle_right + 15:
            return "Left"
        elif angle_right > angle_left + 15:
            return "Right"
        elif nose_tip[1] < left_eye[1] - 20 and nose_tip[1] < right_eye[1] - 20:
            return "Up"
        elif nose_tip[1] > left_eye[1] + 20 and nose_tip[1] > right_eye[1] + 20:
            return "Down"
        else:
            return "Normal"

    def process_face_recognition(self, frame):
        """Run full recognition and direction detection pipeline."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        labeled_frame, labels = self.recognize_face(frame)

        for face in faces:
            shape = self.predictor(gray, face)
            direction = self.detect_face_movement(shape)
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

            cv2.putText(labeled_frame, f"Direction: {direction}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return labeled_frame
