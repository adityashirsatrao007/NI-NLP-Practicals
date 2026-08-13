"""
Expt 4: Lemmatization and Stemming. POS Tagging using Penn Treebank tag set.
"""
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

print("=" * 60)
print("EXPERIMENT 4: Lemmatization, Stemming & POS Tagging")
print("=" * 60)

words = ["running", "flies", "studies", "better", "geese", "happiness",
         "cats", "swimming", "organized", "connection"]

# Stemming
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

print(f"\n{'Word':<15} {'Porter':<15} {'Lancaster':<15} {'Lemmatizer':<15}")
print("-" * 60)
for w in words:
    print(f"{w:<15} {porter.stem(w):<15} {lancaster.stem(w):<15} {lemmatizer.lemmatize(w):<15}")

# POS-based lemmatization
print("\n--- Lemmatization with POS tags ---")
pos_tests = [
    ("running", "v"), ("flies", "n"), ("better", "a"),
    ("geese", "n"), ("running", "n"), ("studies", "n"),
]
for word, pos in pos_tests:
    result = lemmatizer.lemmatize(word, pos=pos)
    print(f"  '{word}' (POS={pos}) -> '{result}'")

# POS Tagging with Penn Treebank tag set
print("\n--- POS Tagging (Penn Treebank) ---")
sentence = "The quick brown fox jumps over the lazy dog near the river bank"
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
print(f"  Sentence: {sentence}")
print(f"  Tagged: {tagged}")

# Tag descriptions
print("\n--- Penn Treebank Tag Set ---")
tag_descriptions = {
    "CC": "Coordinating conjunction", "CD": "Cardinal number",
    "DT": "Determiner", "EX": "Existential there",
    "FW": "Foreign word", "IN": "Preposition/subordinating conjunction",
    "JJ": "Adjective", "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative", "MD": "Modal",
    "NN": "Noun, singular", "NNS": "Noun, plural",
    "NNP": "Proper noun, singular", "RB": "Adverb",
    "RBR": "Adverb, comparative", "TO": "to",
    "VB": "Verb, base form", "VBD": "Verb, past tense",
    "VBG": "Verb, gerund", "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner", "WP": "Wh-pronoun",
}
for word, tag in tagged:
    desc = tag_descriptions.get(tag, "Other")
    print(f"  {word:<12} -> {tag:<5} ({desc})")
