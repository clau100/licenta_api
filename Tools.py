import requests
from bs4 import BeautifulSoup
import re
import pickle

def get_text_from_url(url):
    """Fetch HTML content and extract text."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()  # Clean extra spaces
        return text
    except requests.RequestException:
        return ""


def save_model(model, vectorizer, model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
    """Save model and vectorizer."""
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_model(model_path="model.pkl", vectorizer_path="vectorizer.pkl"):
    """Load model and vectorizer."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None
