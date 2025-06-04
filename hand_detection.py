import cv2
from cvzone.HandTrackingModule import HandDetector

class HandTracking:
    def __init__(self):
        self.detector = HandDetector(detectionCon=0.8)  # Set confidence threshold

    def process_hand_tracking(self, image): # Changed to take image as argument
        image = cv2.flip(image, 1)  # Flip for a mirrored view

        hands, image = self.detector.findHands(image)  # Get hand data
        hand_data = None
        if hands:
            lmlist = hands[0]["lmList"]  # Landmark list
            x, y = lmlist[8][0], lmlist[8][1]  # Index finger tip coordinates
            up = self.detector.fingersUp(hands[0])  # Get which fingers are up
            hand_data = {"fingers_up": up, "index_tip": (x, y)}
        return image, hand_data