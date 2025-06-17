# face_recognition_module.py
import cv2
import numpy as np
import os
import psycopg2
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Database configuration
db_config = {
    "host": "localhost",
    "dbname": "face_recognition",
    "user": "postgres",
    "password": "root"
}

class FaceRecognition:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(320, 320))

        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()

        self.dataset_path = "Training_images"
        self.class_ids, self.embeddings = self.load_dataset()

    def get_student_name(self, student_id):
        try:
            self.cursor.execute("SELECT first_name, last_name FROM students WHERE id = %s", (student_id,))
            result = self.cursor.fetchone()
            if result:
                return f"{result[0]} {result[1]}"
            else:
                return "Unknown"
        except Exception as e:
            print(f"DB Error: {e}")
            return "Unknown"

    def load_dataset(self):
        ids, embeddings = [], []
        for file_name in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = self.app.get(img)
            if faces:
                embedding = faces[0].embedding
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
                ids.append(os.path.splitext(file_name)[0])
        return ids, np.array(embeddings)

    def get_names_for_faces(self, frame):
        faces = self.app.get(frame)
        face_data = []
        for face in faces:
            face_embedding = face.embedding
            face_embedding = face_embedding / np.linalg.norm(face_embedding)
            best_match_id = "Unknown"
            if len(self.embeddings) > 0:
                similarities = cosine_similarity(self.embeddings, face_embedding.reshape(1, -1)).flatten()
                max_index = np.argmax(similarities)
                if similarities[max_index] > 0.3:
                    best_match_id = self.class_ids[max_index]
            name = self.get_student_name(best_match_id)
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_data.append((best_match_id, name, (x1, y1, x2, y2)))
        return face_data
