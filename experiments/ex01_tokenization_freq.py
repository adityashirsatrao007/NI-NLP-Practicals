"""
Expt 1: Convert text into tokens. Find word frequency.
"""
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

text = """
Natural language processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and
human language. The goal is to enable computers to understand, interpret, and
generate human language in a valuable way. NLP combines computational linguistics
with statistical, machine learning, and deep learning models.
"""

print("=" * 60)
print("EXPERIMENT 1: Tokenization & Word Frequency")
print("=" * 60)

# Sentence tokenization
sentences = sent_tokenize(text)
print(f"\nSentence Tokenization ({len(sentences)} sentences):")
for i, s in enumerate(sentences, 1):
    print(f"  [{i}] {s.strip()}")

# Word tokenization
words = word_tokenize(text.lower())
print(f"\nWord Tokenization ({len(words)} tokens):")
print(f"  {words}")

# Word frequency
freq = FreqDist(words)
print(f"\nTop 15 Word Frequencies:")
for word, count in freq.most_common(15):
    print(f"  '{word}': {count}")

# Plot
plt.figure(figsize=(10, 5))
freq.plot(15, title="Word Frequency Distribution", marker="o")
plt.tight_layout()
plt.savefig("outputs/ex01_word_frequency.png", dpi=150)
plt.close()
print("\nPlot saved to outputs/ex01_word_frequency.png")
