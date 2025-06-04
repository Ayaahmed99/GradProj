import cv2
import numpy as np
import math
from mouth_detector import process_mouth  # Keep if mouth detection is needed
from FaceRec import FaceRecognitionMovement

class YOLOv8FaceDetector:
    def __init__(self, model_path, conf_threshold=0.2, iou_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = ['face']
        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16
        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / s), math.ceil(self.input_width / s)) for s in self.strides]
        self.anchors = self._make_anchors(self.feats_hw)

    def _make_anchors(self, feats_hw, offset=0.5):
        anchors = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(w) + offset
            y = np.arange(h) + offset
            sx, sy = np.meshgrid(x, y)
            anchors[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchors

    def _softmax(self, x, axis=1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def _resize_image(self, img, keep_ratio=True):
        top, left = 0, 0
        new_h, new_w = self.input_height, self.input_width

        if keep_ratio and img.shape[0] != img.shape[1]:
            hw_scale = img.shape[0] / img.shape[1]
            if hw_scale > 1:
                new_h = self.input_height
                new_w = int(self.input_width / hw_scale)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                left = (self.input_width - new_w) // 2
                resized = cv2.copyMakeBorder(resized, 0, 0, left, self.input_width - new_w - left,
                                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                new_w = self.input_width
                new_h = int(self.input_height * hw_scale)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                top = (self.input_height - new_h) // 2
                resized = cv2.copyMakeBorder(resized, top, self.input_height - new_h - top, 0, 0,
                                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            resized = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)

        return resized, new_h, new_w, top, left

    def detect(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img, new_h, new_w, pad_top, pad_left = self._resize_image(img_rgb)
        scale_h, scale_w = frame.shape[0] / new_h, frame.shape[1] / new_w
        input_blob = resized_img.astype(np.float32) / 255.0
        blob = cv2.dnn.blobFromImage(input_blob)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        return self._post_process(outputs, scale_h, scale_w, pad_top, pad_left)

    def _post_process(self, outputs, scale_h, scale_w, pad_top, pad_left):
        bboxes, scores, landmarks = [], [], []

        for pred in outputs:
            stride = int(self.input_height / pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))

            box_pred = pred[..., :self.reg_max * 4]
            cls_pred = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))
            kpt_pred = pred[..., -15:].reshape((-1, 15))

            box_pred = self._softmax(box_pred.reshape(-1, 4, self.reg_max), axis=-1)
            box_pred = np.dot(box_pred, self.project).reshape((-1, 4))

            anchors = self.anchors[stride]
            boxes = self._distance2bbox(anchors, box_pred, max_shape=(self.input_height, self.input_width)) * stride

            kpt_pred[:, 0::3] = (kpt_pred[:, 0::3] * 2 + (anchors[:, 0].reshape(-1, 1) - 0.5)) * stride
            kpt_pred[:, 1::3] = (kpt_pred[:, 1::3] * 2 + (anchors[:, 1].reshape(-1, 1) - 0.5)) * stride
            kpt_pred[:, 2::3] = 1 / (1 + np.exp(-kpt_pred[:, 2::3]))

            # Adjust for padding and scale
            boxes -= np.array([[pad_left, pad_top, pad_left, pad_top]])
            boxes *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpt_pred -= np.tile(np.array([pad_left, pad_top, 0]), 5).reshape((1, 15))
            kpt_pred *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1, 15))

            bboxes.append(boxes)
            scores.append(cls_pred)
            landmarks.append(kpt_pred)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)

        # Convert to width-height format for NMS
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]

        confidences = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = confidences > self.conf_threshold
        bboxes_wh, confidences, class_ids, landmarks = bboxes_wh[mask], confidences[mask], class_ids[mask], landmarks[mask]

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold)
        if indices is not None and len(indices) > 0:
            indices = np.array(indices).flatten()
            return bboxes_wh[indices], confidences[indices], class_ids[indices], landmarks[indices]
        return np.array([]), np.array([]), np.array([]), np.array([])

    def _distance2bbox(self, points, distances, max_shape=None):
        x1 = points[:, 0] - distances[:, 0]
        y1 = points[:, 1] - distances[:, 1]
        x2 = points[:, 0] + distances[:, 2]
        y2 = points[:, 1] + distances[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def draw_detections(self, img, boxes, scores, landmarks):
        for box, score, kpt in zip(boxes, scores, landmarks):
            x, y, w, h = box.astype(int)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, f"face:{score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            for i in range(5):
                cx, cy = int(kpt[i * 3]), int(kpt[i * 3 + 1])
                cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
        return img


def process_cropped_face(face_img):
    """Process only mouth region."""
    processed = process_mouth(face_img)
    cv2.imshow("Cropped Face", processed)
    cv2.waitKey(1)


if __name__ == '__main__':
    model_path = r"D:\GradProj\MAINFILES\Models\yolov8n-face.onnx"
    conf_threshold = 0.45
    iou_threshold = 0.5

    face_detector = YOLOv8FaceDetector(model_path, conf_threshold, iou_threshold)
    face_recognition = FaceRecognitionMovement()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores, _, landmarks = face_detector.detect(frame)

        for box in boxes:
            x, y, w, h = box.astype(int)
            x, y = max(0, x), max(0, y)
            w, h = max(1, w), max(1, h)
            cropped = frame[y:y + h, x:x + w]
            process_cropped_face(cropped)

        output_frame, names = face_recognition.recognize_face(frame)
        cv2.imshow("YOLOv8 Face Detection - Webcam", output_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Esc key to quit
            break

    cap.release()
    cv2.destroyAllWindows()
