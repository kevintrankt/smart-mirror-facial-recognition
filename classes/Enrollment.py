import imutils
import cv2
from Camera import Camera

class Enrollment():
    def __init__(self, cascade_file):
      
        # Load Cascade Facial Detection
        self.detector = cv2.CascadeClassifier(cascade_file)

    def detectFace(self, frame):
        """
        Detects face from video stream. Returns Found Status and Resize Frame
        """
        face_detected = True

        # Resize the Frame
        frame = imutils.resize(frame, width=400)
        
        # Grayscale the Frame
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dectect Face in Grayscaled Frame
        facial_area = self.detector.detectMultiScale(
                        grayscale_frame,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )
     
        # No Face Detected
        if facial_area == ():
            face_detected = False

        # Redraw Frame with Outlined Facial Area
        for (x, y, w, h) in facial_area:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return face_detected, frame

    def saveFrame(self, frame, filename):
        return cv2.imwrite(filename, frame)


if __name__ == '__main__':
    cascade_file = "resources/haarcascade_frontalface_default.xml"

    enrollment = Enrollment(cascade_file=cascade_file)

    camera = Camera()
    camera.startVideoStream()
    
    file_count = 0
    while True:
        frame = camera.getFrame()
        detected, drawn_frame = enrollment.detectFace(frame)

        if detected:
            cv2.imshow("Frame", drawn_frame)
        else:
            cv2.imshow("Frame", drawn_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("k"):
            enrollment.saveFrame(frame, "image{}.png".format(file_count))
            file_count += 1
        elif key == ord("q"):
            break

    camera.endVideoStream()

