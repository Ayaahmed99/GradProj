import time
from collections import deque

CHEATING_CONFIDENCE_THRESHOLD_WARNING = 0.8
CHEATING_CONFIDENCE_THRESHOLD_WITHDRAWAL = 0.85
WARNING_FRAME_WINDOW = 3
WITHDRAWAL_TIME_SECONDS = 10
WARNING_LIMIT = 3
CAMERA_OBSTRUCTION_THRESHOLD = 15

class BehaviorMonitor:
    def __init__(self):
        self.cheating_frames = deque(maxlen=WARNING_FRAME_WINDOW)
        self.cheating_start_time = None
        self.warning_count = 0
        self.withdrawn = False
        self.last_seen = time.time()

    def update(self, confidence, label):
        current_time = time.time()
        self.last_seen = current_time

        if label == "cheating" and confidence >= CHEATING_CONFIDENCE_THRESHOLD_WARNING:
            self.cheating_frames.append(1)
        else:
            self.cheating_frames.append(0)

        if sum(self.cheating_frames) == WARNING_FRAME_WINDOW:
            self.issue_warning()

        if label == "cheating" and confidence >= CHEATING_CONFIDENCE_THRESHOLD_WITHDRAWAL:
            if self.cheating_start_time is None:
                self.cheating_start_time = current_time
            elif (current_time - self.cheating_start_time) >= WITHDRAWAL_TIME_SECONDS:
                self.withdraw_student("Sustained cheating behavior")
        else:
            self.cheating_start_time = None

    def issue_warning(self):
        self.warning_count += 1
        print(f"⚠️ Warning {self.warning_count} issued")
        if self.warning_count >= WARNING_LIMIT:
            self.withdraw_student("Exceeded warning limit")

    def withdraw_student(self, reason):
        if not self.withdrawn:
            self.withdrawn = True
            print(f"❌ Student withdrawn: {reason}")

    def check_camera_obstruction(self):
        if time.time() - self.last_seen > CAMERA_OBSTRUCTION_THRESHOLD:
            self.withdraw_student("Camera blackout or obstruction detected")

class MultiPersonMonitor:
    def __init__(self):
        self.persons = {}

    def update(self, box_id, label, confidence):
        if box_id not in self.persons:
            self.persons[box_id] = BehaviorMonitor()

        self.persons[box_id].update(confidence, label)

    def get_status(self, box_id):
        monitor = self.persons.get(box_id)
        if not monitor:
            return "No record"
        return "Withdrawn" if monitor.withdrawn else f"Warnings: {monitor.warning_count}"

    def check_all_obstructions(self):
        for monitor in self.persons.values():
            monitor.check_camera_obstruction()
