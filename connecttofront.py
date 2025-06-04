from flask import Flask, Response
import cv2
from hand_detection import HandTracking
from body_movement import BodyMovementDetector
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

hand_tracker = HandTracking()
body_detector = BodyMovementDetector(
    r"D:\GradProj\Face_Recognition_Attendance_Projects_main\Face_Recognition_Attendance_Projects_main\yolo11n.pt"
)

video_capture = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        processed_frame, _ = hand_tracker.process_hand_tracking(frame.copy())
        processed_frame = body_detector.process_body_movement(processed_frame)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return "✅ Flask Backend Running"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
