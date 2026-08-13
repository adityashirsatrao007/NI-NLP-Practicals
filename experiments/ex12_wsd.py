"""
Expt 12: Word Sense Disambiguation by LSTM/GRU.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, GRU, Dense,
    Bidirectional, Dropout, concatenate, GlobalMaxPooling1D
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

print("=" * 60)
print("EXPERIMENT 12: Word Sense Disambiguation (LSTM/GRU)")
print("=" * 60)

# WSD dataset: ambiguous words with different senses
# "bank" -> financial institution vs river bank
# "light" -> not heavy vs illumination
# "bat" -> animal vs sports equipment
# "crane" -> bird vs machine
# "spring" -> season vs water source vs jump

wsd_data = [
    # bank (sense 0: financial, sense 1: river)
    ("I deposited money at the bank", "bank", 0),
    ("The bank approved my loan application", "bank", 0),
    ("She works at a bank downtown", "bank", 0),
    ("The bank charges high interest rates", "bank", 0),
    ("He invested his savings in the bank", "bank", 0),
    ("The river bank was eroded by floods", "bank", 1),
    ("We sat on the bank of the river", "bank", 1),
    ("Fish were swimming near the bank", "bank", 1),
    ("The bank of the river was muddy", "bank", 1),
    ("She walked along the river bank", "bank", 1),
    # light (sense 0: not heavy, sense 1: illumination)
    ("The bag is very light to carry", "light", 0),
    ("She wore a light dress in summer", "light", 0),
    ("This laptop is surprisingly light", "light", 0),
    ("The feather is light as air", "light", 0),
    ("He picked up the light box easily", "light", 0),
    ("The light from the candle was dim", "light", 1),
    ("Turn on the light in the room", "light", 1),
    ("The sunlight streaming through the window", "light", 1),
    ("She carried a torch light at night", "light", 1),
    ("The light was too bright to look at", "light", 1),
    # bat (sense 0: animal, sense 1: sports equipment)
    ("The bat flew out of the cave at dusk", "bat", 0),
    ("Bats use echolocation to navigate", "bat", 0),
    ("A bat hung upside down from the ceiling", "bat", 0),
    ("The vampire bat is found in tropical regions", "bat", 0),
    ("He hit the ball with the bat", "bat", 1),
    ("The cricket bat is made of willow", "bat", 1),
    ("She swung the bat and scored a six", "bat", 1),
    ("The baseball bat cracked on impact", "bat", 1),
    ("He brought his bat to the game", "bat", 1),
    # crane (sense 0: bird, sense 1: machine)
    ("The crane spread its wings and flew away", "crane", 0),
    ("A beautiful crane stood by the lake", "crane", 0),
    ("The crane is an endangered species", "crane", 0),
    ("Cranes migrate south in winter", "crane", 0),
    ("The construction crane lifted the beam", "crane", 1),
    ("They used a crane to build the bridge", "crane", 1),
    ("The crane operator worked from dawn to dusk", "crane", 1),
    ("A giant crane towered over the construction site", "crane", 1),
]

# Build vocabulary
all_words = set()
for sent, _, _ in wsd_data:
    for w in sent.split():
        all_words.add(w.lower())

word2idx = {w: i + 2 for i, w in enumerate(sorted(all_words))}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1
VOCAB_SIZE = len(word2idx)
MAX_LEN = 20

# Prepare data
X = []
y = []
for sent, target_word, sense in wsd_data:
    words = sent.lower().split()
    x_seq = [word2idx.get(w, 1) for w in words]
    X.append(x_seq)
    y.append(sense)

X = pad_sequences(X, maxlen=MAX_LEN, padding="post")
y = to_categorical(y, num_classes=2)

print(f"Dataset: {len(X)} examples, {VOCAB_SIZE} vocabulary")
print(f"Senses: 2 (0=first meaning, 1=second meaning)")

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Model 1: LSTM ──
print("\n=== Model 1: Bidirectional LSTM ===")
input_layer = Input(shape=(MAX_LEN,))
embedding = Embedding(VOCAB_SIZE, 64, mask_zero=True)(input_layer)
lstm_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(embedding)
pooled = GlobalMaxPooling1D()(lstm_out)
dense = Dense(32, activation="relu")(pooled)
dropout = Dropout(0.3)(dense)
output = Dense(2, activation="softmax")(dropout)

lstm_model = Model(inputs=input_layer, outputs=output)
lstm_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history_lstm = lstm_model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=0)
lstm_acc = lstm_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"LSTM Test Accuracy: {lstm_acc:.2%}")

# ── Model 2: GRU ──
print("\n=== Model 2: Bidirectional GRU ===")
input_layer2 = Input(shape=(MAX_LEN,))
embedding2 = Embedding(VOCAB_SIZE, 64, mask_zero=True)(input_layer2)
gru_out = Bidirectional(GRU(64, return_sequences=True, dropout=0.2))(embedding2)
pooled2 = GlobalMaxPooling1D()(gru_out)
dense2 = Dense(32, activation="relu")(pooled2)
dropout2 = Dropout(0.3)(dense2)
output2 = Dense(2, activation="softmax")(dropout2)

gru_model = Model(inputs=input_layer2, outputs=output2)
gru_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history_gru = gru_model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=0)
gru_acc = gru_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"GRU Test Accuracy: {gru_acc:.2%}")

# ── Predictions ──
print("\n--- Predictions on New Sentences ---")
test_sentences = [
    ("I need to go to the bank to withdraw cash", "bank"),
    ("The bank of the river was full of rocks", "bank"),
    ("Please turn on the light", "light"),
    ("This package is very light", "light"),
    ("The crane flew over the wetland", "crane"),
    ("The crane lifted the heavy steel beam", "crane"),
]

for sent, target in test_sentences:
    words = sent.lower().split()
    x_test = [word2idx.get(w, 1) for w in words]
    x_test = pad_sequences([x_test], maxlen=MAX_LEN, padding="post")

    lstm_pred = lstm_model.predict(x_test, verbose=0)
    gru_pred = gru_model.predict(x_test, verbose=0)

    lstm_sense = np.argmax(lstm_pred[0])
    gru_sense = np.argmax(gru_pred[0])
    lstm_conf = np.max(lstm_pred[0]) * 100
    gru_conf = np.max(gru_pred[0]) * 100

    sense_labels = {"bank": ["financial", "river"], "light": ["lightweight", "illumination"],
                    "crane": ["bird", "machine"]}

    lstm_label = sense_labels[target][lstm_sense]
    gru_label = sense_labels[target][gru_sense]

    print(f"\n  '{sent}'")
    print(f"  Target word: '{target}'")
    print(f"  LSTM: {lstm_label} (conf: {lstm_conf:.1f}%)")
    print(f"  GRU:  {gru_label} (conf: {gru_conf:.1f}%)")

print(f"\n--- Comparison ---")
print(f"  LSTM accuracy: {lstm_acc:.2%}")
print(f"  GRU  accuracy: {gru_acc:.2%}")
