import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import pickle
import requests
from bs4 import BeautifulSoup

# --- CONFIG ---
MODEL_NAME = "FacebookAI/xlm-roberta-base"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEADERS = {'User-Agent': 'Mozilla/5.0'}

# --- Helper: Fetch and clean live HTML ---
def fetch_and_clean_html(url):
    try:
        response = requests.get(
            f"http://{url}",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
            },
            timeout=10
        )
        if response.status_code != 200:
            print(f"⚠️ {url} returned {response.status_code}")
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return ""


# --- Dataset class ---
class HTMLDataset(Dataset):
    def __init__(self, urls, labels, tokenizer, max_len):
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        text = fetch_and_clean_html(self.urls[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# --- Load CSV with url + label ---
df = pd.read_csv("html_dataset.csv")  # Columns: url, label
urls = df['url'].tolist()
labels = df['label'].tolist()

X_train, X_val, y_train, y_val = train_test_split(urls, labels, test_size=0.2, stratify=labels)

# --- Tokenizer and datasets ---
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
train_dataset = HTMLDataset(X_train, y_train, tokenizer, MAX_LEN)
val_dataset = HTMLDataset(X_val, y_val, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- Load model ---
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

# --- Training loop ---
model.train()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- Evaluation ---
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

# --- Save model and tokenizer ---
model.save_pretrained("bert_html_model_live")
tokenizer.save_pretrained("bert_html_model_live")

# --- Save predictions + labels with pickle ---
with open("bert_html_results_live.pkl", "wb") as f:
    pickle.dump({
        "predictions": all_preds,
        "labels": all_labels
    }, f)

print("\n✅ Done! Model trained with live HTML fetched from the web.")
