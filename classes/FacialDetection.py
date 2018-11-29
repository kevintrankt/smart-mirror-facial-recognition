import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os


from Camera import Camera

class FacialDetection():
    def __init__(self, proto_path, model_path, embedding_model, recognizer_file, label_encoder_file):
        # Load Detector
        self.detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

        # Load Embedder
        self.embedder = cv2.dnn.readNetFromTorch(embedding_model)

        # load the actual face recognition model along with the label encoder
        self.recognizer = pickle.loads(open(recognizer_file, "rb").read())
        self.label_encoder = pickle.loads(open(label_encoder_file, "rb").read())

    def detectFace(self, frame):
        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions

        # Resize the Frame
        frame = imutils.resize(frame, width=600)
        
        # Get the Dimensions of the Frame
        (h, w) = frame.shape[:2]

        # Resize the Image
        image = cv2.resize(frame, (300, 300))

        # Construct a Blob From the Image
        image_blob = cv2.dnn.blobFromImage(
                                    image,
                                    scalefactor=1.0,
                                    size=(300, 300),
                                    mean=(104.0, 177.0, 123.0),
                                    swapRB=False,
                                    crop=False
                                )

        # Localize the Faces From the Input Image
        self.detector.setInput(image_blob)
        detections = self.detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            
            # Get the Confidence of the Face
            confidence = detections[0, 0, i, 2]

            # Ignore Weak Confidence
            if confidence > 0.5:
                # Find the coordinates of the Box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x_start, y_start, x_end, y_end) = box.astype("int")

                # Get the Face
                face       = image[y_start:y_end, x_start:x_end]
                (f_h, f_w) = face.shape[:2]

                # Skip Faces that are Too Small
                if f_h < 20 or f_w < 20:
                    continue

                face_blob = cv2.dnn.blobFromImage(
                                            face,
                                            scalefactor=1.0 / 255,
                                            size=(96, 96),
                                            mean=(0, 0, 0),
                                            swapRB=True,
                                            crop=False
                                        )
                # Set the Embedder and Generate the Vector
                self.embedder.setInput(face_blob)
                vector = self.embedder.forward()

                # Classify the Face
                prediction = self.recognizer.predict_proba(vector)[0]
                j           = np.argmax(prediction)
                probability = prediction[j]
                user        = self.label_encoder.classes_[j]

                # Classification Text
                text = "{}: {:.2f}%".format(user, probability * 100)

                # Draw Rectangle on Face and Show Text
                y = y_start - 10 if y_start - 10 > 10 else y_start + 10
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
                cv2.putText(frame, text, (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                return frame

if __name__ == '__main__':
    proto_path      = "resources/deploy.prototxt"
    model_path      = "resources/res10_300x300_ssd_iter_140000.caffemodel"
    embedding_model = "resources/openface.nn4.small2.v1.t7"
    
    recognizer_file    = "output/recognizer.pickle"
    label_encoder_file = "output/label_encoder.pickle"

    facial_dectection = FacialDetection(proto_path, model_path, embedding_model, recognizer_file, label_encoder_file)
    
    camera = Camera()
    camera.startVideoStream()

    while True:
        frame = camera.getFrame()
        drawn_frame = facial_dectection.detectFace(frame)

        cv2.imshow("Frame", drawn_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.endVideoStream()