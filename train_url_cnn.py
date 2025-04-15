import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# 📥 Load from CSV
df = pd.read_csv("sites.csv")

# ✅ Use only URL and label for now
urls = df["url"].astype(str).tolist()
labels = df["label"].tolist()

# 🔢 Tokenize URLs (char-level)
max_len = 50
vocab_size = 1000

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(urls)
sequences = tokenizer.texts_to_sequences(urls)
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)

# 📚 Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 CNN Model
input_layer = Input(shape=(max_len,))
embedding = Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len)(input_layer)
conv = Conv1D(filters=64, kernel_size=5, activation='relu')(embedding)
pool = GlobalMaxPooling1D()(conv)
dense = Dense(64, activation='relu')(pool)
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🚀 Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# 💾 Save model and tokenizer
model.save("url_cnn_model.h5")
with open("url_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ URL CNN training complete.")
