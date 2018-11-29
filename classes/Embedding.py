import os
import cv2
import pickle
import numpy as np
import imutils
from imutils import paths

class Embeddings():
    def __init__(self, proto_path, model_path, embedding_model):
        # Load Detector
        self.detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

        # Load Serialized Face Embedding Model
        self.embedder = cv2.dnn.readNetFromTorch(embedding_model)

    def createEmbedding(self, dataset, embeddings_file):
        users      = []
        embeddings = []

        # Load Images
        image_paths = list(paths.list_images(dataset))

        for (i, image_path) in enumerate(image_paths):
            # Get the Image
            user = image_path.split(os.path.sep)[-2]

            # Load the Image
            image = cv2.imread(image_path)
            image = imutils.resize(image, width=600)

            # Get the Dimenions of the Image
            (h, w) = image.shape[:2]

            # Resize the Image
            image = cv2.resize(image, dsize=(300, 300))

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

            # Find the Largest Box Face
            i = np.argmax(detections[0, 0, :, 2])

            # Get the Confidence of the Face
            confidence = detections[0, 0, i, 2]

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

            users.append(user)
            embeddings.append(vector.flatten())

        # Create a Dictionary of the Users and Embeddings
        data = {
            "users": users,
            "embeddings": embeddings
        }

        # Output to Pickle File
        with open(embeddings_file,"wb") as file:
            file.write(pickle.dumps(data))

if __name__ == '__main__':
    proto_path      = "resources/deploy.prototxt"
    model_path      = "resources/res10_300x300_ssd_iter_140000.caffemodel"
    embedding_model = "resources/openface.nn4.small2.v1.t7"
    dataset         = "dataset"
    embeddings_file = "output/embeddings.pickle"

    embedding = Embeddings(proto_path=proto_path, model_path=model_path, embedding_model=embedding_model)
    embedding.createEmbedding(dataset, embeddings_file)

