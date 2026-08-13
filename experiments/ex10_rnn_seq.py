"""
Expt 10: Implement RNN for sequence labelling.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

print("=" * 60)
print("EXPERIMENT 10: RNN for Sequence Labelling (POS Tagging)")
print("=" * 60)

# Simple POS tagging dataset
training_data = [
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
    (["The", "big", "brown", "fox", "jumped"],
     ["DT", "JJ", "JJ", "NN", "VBD"]),
    (["She", "gave", "him", "a", "book"],
     ["PRP", "VBD", "PRP", "DT", "NN"]),
    (["They", "played", "soccer", "yesterday"],
     ["PRP", "VBD", "NN", "NN"]),
    (["The", "student", "studied", "hard", "for", "the", "exam"],
     ["DT", "NN", "VBD", "RB", "IN", "DT", "NN"]),
    (["I", "ate", "a", "delicious", "meal"],
     ["PRP", "VBD", "DT", "JJ", "NN"]),
    (["The", "car", "is", "fast"],
     ["DT", "NN", "VBZ", "JJ"]),
    (["Birds", "fly", "in", "the", "sky"],
     ["NNS", "VBP", "IN", "DT", "NN"]),
    (["She", "reads", "books", "every", "day"],
     ["PRP", "VBZ", "NNS", "DT", "NN"]),
    (["The", "teacher", "explained", "the", "concept"],
     ["DT", "NN", "VBD", "DT", "NN"]),
]

# Build vocabularies
word_tags = set()
all_words = set()
all_tags = set()
for words, tags in training_data:
    for w, t in zip(words, tags):
        all_words.add(w)
        all_tags.add(t)
        word_tags.add((w, t))

word2idx = {w: i + 2 for i, w in enumerate(sorted(all_words))}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1
tag2idx = {t: i for i, t in enumerate(sorted(all_tags))}
idx2tag = {i: t for t, i in tag2idx.items()}

VOCAB_SIZE = len(word2idx)
NUM_TAGS = len(tag2idx)
MAX_LEN = max(len(words) for words, _ in training_data)

print(f"Vocabulary: {VOCAB_SIZE} words")
print(f"Tags: {NUM_TAGS} ({list(sorted(all_tags))})")
print(f"Max sequence length: {MAX_LEN}")

# Prepare data
X = []
y = []
for words, tags in training_data:
    x_seq = [word2idx.get(w, 1) for w in words]
    y_seq = [tag2idx[t] for t in tags]
    X.append(x_seq)
    y.append(y_seq)

X = pad_sequences(X, maxlen=MAX_LEN, padding="post")
y = pad_sequences(y, maxlen=MAX_LEN, padding="post")
y_cat = to_categorical(y, num_classes=NUM_TAGS)

print(f"X shape: {X.shape}")
print(f"y shape: {y_cat.shape}")

# Build RNN model
model = Sequential([
    Embedding(VOCAB_SIZE, 32, input_length=MAX_LEN),
    SimpleRNN(64, return_sequences=True),
    TimeDistributed(Dense(NUM_TAGS, activation="softmax"))
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
print("\n--- Training ---")
history = model.fit(X, y_cat, epochs=200, verbose=0)

final_acc = history.history["accuracy"][-1]
final_loss = history.history["loss"][-1]
print(f"Final accuracy: {final_acc:.2%}")
print(f"Final loss: {final_loss:.4f}")

# Predict
print("\n--- Predictions ---")
test_sentences = [
    ["The", "cat", "sat", "on", "the", "mat"],
    ["She", "runs", "every", "morning"],
    ["I", "love", "natural", "language"],
]

for sent in test_sentences:
    x_test = [word2idx.get(w, 1) for w in sent]
    x_test = pad_sequences([x_test], maxlen=MAX_LEN, padding="post")
    pred = model.predict(x_test, verbose=0)
    pred_tags = [idx2tag[np.argmax(p)] for p in pred[0][:len(sent)]]
    print(f"\n  Words: {sent}")
    print(f"  Predicted POS: {pred_tags}")
    print(f"  True POS:      {[training_data[0][1][i] if i < len(training_data[0][1]) else '?' for i in range(len(sent))]}")
