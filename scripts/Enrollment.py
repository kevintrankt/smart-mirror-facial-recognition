from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

import logging, sys

class Enrollment():
    def __init__(self, cascade_file, RPI3=False, show_video_feed=False, level=logging.CRITICAL):
        self.RPI3            = RPI3
        self.show_video_feed = show_video_feed

        logging.basicConfig(stream=sys.stderr, level=level)
        
        # Load Cascade Facial Detection
        self.detector = cv2.CascadeClassifier(cascade_file)
        
    def startVideoStream(self):
        logging.info("Starting video stream")
        if self.RPI3:
            # User Raspberry Pi3 Camera Module
            self.video_stream = VideoStream(usePiCamera=True).start()
        else: 
            # Use Default Camera
            self.video_stream = VideoStream(src=0).start()

        # Allow for the Camera to Wake
        time.sleep(2.0)

    def endVideoStream(self):
        cv2.destroyAllWindows()
        self.video_stream.stop()

    def detectFace(self):
        """
        Detects face from video stream. Returns Original Frame if Found, None if not detected
        """

        # Extract the Frame from the Video Stream
        frame = self.video_stream.read()

        # Resize the Frame
        resized_frame = imutils.resize(frame, width=400)
        
        # Grayscale the Frame
        grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Dectect Face in Grayscaled Frame
        facial_area = self.detector.detectMultiScale(
                        grayscale_frame,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )
     
        # No Face Detected, Return From Function
        if facial_area == ():
            logging.debug("No face detected")
            return None

        # Redraw Frame with Outlined Facial Area
        for (x, y, w, h) in facial_area:
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the Output Frame
        if self.show_video_feed:
            cv2.imshow("Frame", resized_frame)

        return frame

    def saveFrame(self, frame, filename):
        return cv2.imwrite(filename, frame)


if __name__ == '__main__':
    cascade_file = "resources/haarcascade_frontalface_default.xml"

    enrollment = Enrollment(cascade_file=cascade_file, show_video_feed=True, level=logging.DEBUG)
    enrollment.startVideoStream()
    
    file_count = 0
    while True:
        frame = enrollment.detectFace()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("k"):
            enrollment.saveFrame(frame, "image{}.png".format(file_count))
            file_count += 1
        elif key == ord("q"):
            break

    enrollment.endVideoStream()

