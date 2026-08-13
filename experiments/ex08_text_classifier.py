"""
Expt 8: Implement text classifier using logistic regression model.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

print("=" * 60)
print("EXPERIMENT 8: Text Classifier (Logistic Regression)")
print("=" * 60)

# Dataset: news topic classification
texts = [
    # Sports
    "The team won the championship game in overtime",
    "The player scored a hat trick in the football match",
    "The Olympics will be held in Paris next year",
    "The tennis player won the Grand Slam title",
    "The basketball team drafted a new star player",
    "The cricket match ended in a draw on the final day",
    "The soccer coach was fired after losing streak",
    "The swimmer broke the world record in the 100m freestyle",
    "The boxing match was declared a knockout victory",
    "The rugby team scored a try in the last minute",
    # Technology
    "The new iPhone has an improved camera and processor",
    "Artificial intelligence is transforming healthcare industry",
    "The startup raised 50 million dollars in funding",
    "Google released a new update for Android operating system",
    "The cloud computing market is growing rapidly",
    "Machine learning models are being used for fraud detection",
    "The cryptocurrency market crashed overnight",
    "Quantum computing breakthrough achieved by researchers",
    "The robot uses deep learning for navigation",
    "Blockchain technology is revolutionizing supply chain",
    # Politics
    "The president signed the new trade agreement today",
    "The senator proposed a new healthcare reform bill",
    "Elections will be held in November across the country",
    "The government announced new environmental policies",
    "The prime minister addressed the parliament on budget",
    "New legislation aims to regulate social media platforms",
    "The diplomat met with foreign minister for peace talks",
    "The opposition party criticized the ruling government",
    "Voter turnout was highest in the last decade",
    "The congress passed the infrastructure spending bill",
    # Science
    "Scientists discovered a new species in the Amazon rainforest",
    "The Mars rover found evidence of ancient water",
    "Researchers developed a new vaccine for tropical disease",
    "The telescope captured images of distant galaxies",
    "Climate change is causing glaciers to melt faster",
    "The experiment confirmed the theory of quantum entanglement",
    "Biologists mapped the complete genome of a new organism",
    "The particle accelerator detected a new subatomic particle",
    "Astronomers found an exoplanet in the habitable zone",
    "The clinical trial showed promising results for cancer treatment",
]

labels = (
    ["Sports"] * 10 +
    ["Technology"] * 10 +
    ["Politics"] * 10 +
    ["Science"] * 10
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n--- Model Performance ---")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict new texts
print("\n--- Predictions on New Texts ---")
new_texts = [
    "The quarterback threw a touchdown pass",
    "Apple announced the new MacBook Pro",
    "The election results were announced today",
    "Researchers found a cure for the disease",
    "The stock market hit record highs",
]
new_X = vectorizer.transform(new_texts)
predictions = model.predict(new_X)
for text, pred in zip(new_texts, predictions):
    print(f"  '{text}' -> {pred}")

# Feature importance
print("\n--- Top Features per Class ---")
feature_names = vectorizer.get_feature_names_out()
for i, cls in enumerate(model.classes_):
    top_idx = np.argsort(model.coef_[i])[-5:][::-1]
    top_features = [feature_names[j] for j in top_idx]
    print(f"  {cls}: {top_features}")
