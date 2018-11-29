import cv2
import time
from imutils.video import VideoStream

class Camera():
    def __init__(self, RPI3=False):
        self.RPI3            = RPI3
        self.video_stream    = None

    def startVideoStream(self):
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

    def getFrame(self):
        return self.video_stream.read()