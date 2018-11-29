from Camera          import Camera
from Embedding       import Embedding
from Enrollment      import Enrollment
from FacialDetection import FacialDetection
from TrainModel      import TrainModel

from pynput.keyboard import Key, Controller

if __name__ == '__main__':
    def signIn():
        global logged_in
        # Wait to Detect Face
        while True:
            frame = camera.getFrame()
            user  = facial_detection.detectFace(frame)
            if user != "unknown":
                # If user is known, trigger their profile
                keyboard.type(string(user))
                break
        logged_in = True

    def signUp():
        global logged_in

        # Get User ID
        while True:
            frame  = camera.getFrame()
            number = gesture.detectNumber(frame)
            if 1 <= number <= 5:
                break
        # Collect 5 pictures
        count = 0
        while count != 5:
            frame  = camera.getFrame()
            action = gesture.detectGesture(frame)
            detected, drawn_frame  = enrollment.detectFace(frame)
            if action == "capture" and detected:
                enrollment.saveFrame(frame, "dataset/{}/image{}.png".format(user, count))
                count += 1

        logged_in = True
        keyboard.type(string(number))

    def signOut():
        global logged_in
        keyboard.type(Key.space)
        logged_in = False

    cascade_file = "resources/haarcascade_frontalface_default.xml"

    dataset         = "dataset"
    proto_path      = "resources/deploy.prototxt"
    model_path      = "resources/res10_300x300_ssd_iter_140000.caffemodel"
    embedding_model = "resources/openface.nn4.small2.v1.t7"
    
    embeddings_file = "output/embeddings.pickle"
    recognizer_file    = "output/recognizer.pickle"
    label_encoder_file = "output/label_encoder.pickle"

    camera = Camera()
    camera.startVideoStream()

    gesture           = Gesture(cascade_file)
    keyboard          = Controller()
    embedding         = Embeddings(proto_path, model_path, embedding_model)
    enrollment        = Enrollment(cascade_file)
    facial_dectection = FacialDetection(proto_path, model_path, embedding_model, recognizer_file, label_encoder_file)
    train_model       = TrainModel(embeddings_file, recognizer_file, label_encoder_file)
    
    logged_in = False

    while True:
        frame  = camera.getFrame()
        action = gesture.detectGesture(frame)
        if not logged_in:
            if action == "sign_in":
                signIn()
            elif action == "sign_up":
                signUp()
        else:
            if action == "sign_out":
                signOut()
                
    camera.endVideoStream()