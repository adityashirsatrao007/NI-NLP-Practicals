"""
Expt 3: Bigram / Trigram Language Model. Generate regex for given text.
"""
import nltk
from nltk import bigrams, trigrams, FreqDist
from nltk.probability import ConditionalFreqDist
import re

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

text = """
The cat sat on the mat. The dog sat on the log. The cat chased the dog.
The dog chased the cat. The mat was on the floor. The log was on the grass.
"""

print("=" * 60)
print("EXPERIMENT 3: Bigram/Trigram Language Model & Regex")
print("=" * 60)

tokens = nltk.word_tokenize(text.lower())
tokens = [t for t in tokens if t.isalpha()]

# Bigrams
print("\n--- Bigrams ---")
bg = list(bigrams(tokens))
print(f"Total bigrams: {len(bg)}")
cfd_bigram = ConditionalFreqDist(bg)
print("Top bigrams:")
bg_freq = FreqDist(bg)
for (w1, w2), c in bg_freq.most_common(10):
    print(f"  ('{w1}', '{w2}'): {c}")

# Trigrams
print("\n--- Trigrams ---")
tg = list(trigrams(tokens))
print(f"Total trigrams: {len(tg)}")
tg_freq = FreqDist(tg)
print("Top trigrams:")
for (w1, w2, w3), c in tg_freq.most_common(10):
    print(f"  ('{w1}', '{w2}', '{w3}'): {c}")

# Conditional frequency: given a word, what follows?
print("\nConditional FreqDist (word -> next word):")
for word in ["the", "cat", "dog"]:
    if word in cfd_bigram:
        top = cfd_bigram[word].most_common(5)
        print(f"  After '{word}': {top}")

# Regex generation
print("\n--- Regex Patterns ---")
patterns = {
    "Email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "Phone (Indian)": r"\+?91[\s-]?\d{10}|\d{5}[\s-]\d{5}",
    "Date (DD/MM/YYYY)": r"\d{2}/\d{2}/\d{4}",
    "URL": r"https?://(?:www\.)?\S+",
    "IP Address": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
}

samples = {
    "Email": "Contact us at support@example.com or admin@company.org",
    "Phone (Indian)": "Call +919876543210 or 11234567890",
    "Date (DD/MM/YYYY)": "Event on 25/12/2026 and 01/01/2027",
    "URL": "Visit https://www.example.com or http://test.org/page",
    "IP Address": "Server at 192.168.1.1 and 10.0.0.255",
}

for name, pattern in patterns.items():
    matches = re.findall(pattern, samples[name])
    print(f"  {name}: {matches}")
