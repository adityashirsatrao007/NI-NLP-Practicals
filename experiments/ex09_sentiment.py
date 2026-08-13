"""
Expt 9: Implement a movie reviews sentiment classifier.
"""
import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import random

nltk.download("movie_reviews", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

print("=" * 60)
print("EXPERIMENT 9: Movie Reviews Sentiment Classifier")
print("=" * 60)

# Load movie reviews dataset
documents = []
labels = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        doc = " ".join(movie_reviews.words(fileid))
        documents.append(doc)
        labels.append(1 if category == "pos" else 0)

print(f"Dataset: {len(documents)} reviews ({labels.count(1)} positive, {labels.count(0)} negative)")

# Shuffle
combined = list(zip(documents, labels))
random.shuffle(combined)
documents, labels = zip(*combined)

# TF-IDF
stop_words = set(stopwords.words("english"))
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
X = vectorizer.fit_transform(documents)
y = list(labels)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\n--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Predict custom reviews
print("--- Custom Review Predictions ---")
custom_reviews = [
    "This movie was absolutely fantastic! Great acting and storyline.",
    "Terrible film. Waste of time and money. The plot was boring.",
    "An okay movie. Not great but not terrible either.",
    "One of the best movies I have ever seen. Highly recommended!",
    "The acting was wooden and the dialogue was cringe-worthy.",
    "A masterpiece of cinema with brilliant performances.",
    "I fell asleep halfway through. Very dull and predictable.",
    "Stunning visuals and a gripping narrative from start to finish.",
]

custom_X = vectorizer.transform(custom_reviews)
predictions = model.predict(custom_X)
probabilities = model.predict_proba(custom_X)

for review, pred, prob in zip(custom_reviews, predictions, probabilities):
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    confidence = max(prob) * 100
    print(f"\n  Review: '{review[:60]}...'")
    print(f"  Sentiment: {sentiment} (confidence: {confidence:.1f}%)")

# Top discriminative features
print("\n--- Top Features ---")
feature_names = vectorizer.get_feature_names_out()
for i, name in enumerate(model.classes_):
    if i < model.coef_.shape[0]:
        coef = model.coef_[i]
        top_idx = coef.argsort()[-10:][::-1]
        top_words = [feature_names[j] for j in top_idx]
        print(f"  {name}: {top_words}")
