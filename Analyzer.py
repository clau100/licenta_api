from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

class GamblingAnalyzer:
    def __init__(self):
        # Load trained CNN model and tokenizer
        self.model = load_model("url_cnn_model.h5")
        with open("tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        # This must match the max_len used during training
        self.max_len = 50

    def preprocess_url(self, url):
        # Convert the URL to a padded sequence using the same tokenizer
        sequence = self.tokenizer.texts_to_sequences([url])
        padded = pad_sequences(sequence, maxlen=self.max_len)
        return padded

    def predict(self, url):
        try:
            x = self.preprocess_url(url)
            prediction = self.model.predict(x, verbose=0)[0][0]  # probability
            return int(prediction >= 0.5)  # threshold at 0.5
        except Exception as e:
            print(f"[Analyzer Error] Failed to analyze URL '{url}': {e}")
            return 0  # default to not blocking if something goes wrong
