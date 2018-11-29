from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle


class TrainModel():
    def __init__(self, embeddings_file, recognizer_file, label_encoder_file):
        # Reconstruct the Data
        data = pickle.loads(open(embeddings_file, "rb").read())
        
        # Generate the Label Encoder
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(data["users"])

        # Train the Recognizer
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)

        # Write the Recognizer File
        with open(recognizer_file, "wb") as file:
            file.write(pickle.dumps(recognizer))
            file.close()

        with open(label_encoder_file, "wb") as file:
            file.write(pickle.dumps(label_encoder))

if __name__ == '__main__':
    embeddings_file    = "output/embeddings.pickle" 
    recognizer_file    = "output/recognizer.pickle"
    label_encoder_file = "output/label_encoder.pickle"

    train_model = TrainModel(embeddings_file, recognizer_file, label_encoder_file)
