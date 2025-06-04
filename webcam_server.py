import cv2
import multiprocessing

def webcam_stream(frame_queue):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing webcam")
            break
        frame_queue.put(frame)  # Put frame into the queue
    cap.release()

if __name__ == "__main__":
    frame_queue = multiprocessing.Queue()
    webcam_process = multiprocessing.Process(target=webcam_stream, args=(frame_queue,))
    webcam_process.start()
