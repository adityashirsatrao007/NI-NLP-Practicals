"""
Expt 11: POS tagging using LSTM.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

print("=" * 60)
print("EXPERIMENT 11: POS Tagging using LSTM")
print("=" * 60)

# Penn Treebank-like dataset
sentences_data = [
    (["The", "cat", "sat", "on", "the", "mat"],
     ["DT", "NN", "VBD", "IN", "DT", "NN"]),
    (["I", "love", "natural", "language", "processing"],
     ["PRP", "VBP", "JJ", "NN", "NN"]),
    (["She", "runs", "every", "morning"],
     ["PRP", "VBZ", "DT", "NN"]),
    (["The", "dog", "chased", "the", "cat", "quickly"],
     ["DT", "NN", "VBD", "DT", "NN", "RB"]),
    (["We", "are", "learning", "deep", "learning"],
     ["PRP", "VBP", "VBG", "JJ", "NN"]),
    (["He", "will", "arrive", "tomorrow"],
     ["PRP", "MD", "VB", "NN"]),
    (["The", "big", "brown", "fox", "jumped", "over", "the", "lazy", "dog"],
     ["DT", "JJ", "JJ", "NN", "VBD", "IN", "DT", "JJ", "NN"]),
    (["She", "gave", "him", "a", "beautiful", "book"],
     ["PRP", "VBD", "PRP", "DT", "JJ", "NN"]),
    (["They", "played", "soccer", "yesterday"],
     ["PRP", "VBD", "NN", "NN"]),
    (["The", "student", "studied", "hard", "for", "the", "exam"],
     ["DT", "NN", "VBD", "RB", "IN", "DT", "NN"]),
    (["I", "ate", "a", "delicious", "meal"],
     ["PRP", "VBD", "DT", "JJ", "NN"]),
    (["The", "car", "is", "very", "fast"],
     ["DT", "NN", "VBZ", "RB", "JJ"]),
    (["Birds", "fly", "high", "in", "the", "sky"],
     ["NNS", "VBP", "RB", "IN", "DT", "NN"]),
    (["She", "reads", "many", "books", "every", "day"],
     ["PRP", "VBZ", "JJ", "NNS", "DT", "NN"]),
    (["The", "teacher", "explained", "the", "concept", "clearly"],
     ["DT", "NN", "VBD", "DT", "NN", "RB"]),
    (["A", "small", "cat", "sleeps", "on", "the", "warm", "bed"],
     ["DT", "JJ", "NN", "VBZ", "IN", "DT", "JJ", "NN"]),
    (["The", "children", "played", "in", "the", "park"],
     ["DT", "NNS", "VBD", "IN", "DT", "NN"]),
    (["My", "friend", "works", "at", "a", "technology", "company"],
     ["PRP$", "NN", "VBZ", "IN", "DT", "NN", "NN"]),
    (["The", "old", "man", "walked", "slowly"],
     ["DT", "JJ", "NN", "VBD", "RB"]),
    (["We", "should", "protect", "the", "environment"],
     ["PRP", "MD", "VB", "DT", "NN"]),
]

# Augment data by generating variations
augmented = []
for words, tags in sentences_data:
    augmented.append((words, tags))
    # Slight variations
    if len(words) > 3:
        augmented.append((words[:len(words)//2], tags[:len(tags)//2]))
        augmented.append((words[len(words)//2:], tags[len(tags)//2:]))

sentences_data = augmented

# Build vocabularies
all_words = set()
all_tags = set()
for words, tags in sentences_data:
    for w, t in zip(words, tags):
        all_words.add(w)
        all_tags.add(t)

word2idx = {w: i + 2 for i, w in enumerate(sorted(all_words))}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1
tag2idx = {t: i + 1 for i, t in enumerate(sorted(all_tags))}
tag2idx["<PAD>"] = 0
idx2tag = {i: t for t, i in tag2idx.items()}

VOCAB_SIZE = len(word2idx)
NUM_TAGS = len(tag2idx)
MAX_LEN = max(len(words) for words, _ in sentences_data)

print(f"Vocabulary: {VOCAB_SIZE} words")
print(f"Tags: {NUM_TAGS - 1} ({sorted(all_tags)})")
print(f"Max length: {MAX_LEN}")

# Prepare sequences
X = []
y = []
for words, tags in sentences_data:
    x_seq = [word2idx.get(w, 1) for w in words]
    y_seq = [tag2idx[t] for t in tags]
    X.append(x_seq)
    y.append(y_seq)

X = pad_sequences(X, maxlen=MAX_LEN, padding="post")
y = pad_sequences(y, maxlen=MAX_LEN, padding="post")
y_cat = to_categorical(y, num_classes=NUM_TAGS)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}")

# Bidirectional LSTM model
input_layer = Input(shape=(MAX_LEN,))
embedding = Embedding(VOCAB_SIZE, 64, mask_zero=True)(input_layer)
lstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(embedding)
output = TimeDistributed(Dense(NUM_TAGS, activation="softmax"))(lstm)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train
print("\n--- Training BiLSTM ---")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,
    verbose=0
)

final_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]
print(f"Training accuracy:   {final_acc:.2%}")
print(f"Validation accuracy: {val_acc:.2%}")

# Predict
print("\n--- Predictions ---")
test_sents = [
    ["The", "cat", "sat", "on", "the", "mat"],
    ["She", "runs", "every", "morning"],
    ["I", "love", "deep", "learning"],
    ["The", "big", "brown", "fox", "jumped"],
]

for sent in test_sents:
    x_test = [word2idx.get(w, 1) for w in sent]
    x_test = pad_sequences([x_test], maxlen=MAX_LEN, padding="post")
    pred = model.predict(x_test, verbose=0)
    pred_tags = [idx2tag[np.argmax(p)] for p in pred[0][:len(sent)]]
    print(f"\n  Words: {sent}")
    print(f"  LSTM POS: {pred_tags}")
