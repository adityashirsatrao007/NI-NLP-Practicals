"""
Expt 2: Find synonym / antonym of a word using WordNet.
"""
import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

print("=" * 60)
print("EXPERIMENT 2: Synonyms & Antonyms using WordNet")
print("=" * 60)

words = ["good", "happy", "fast", "big", "smart"]

for word in words:
    synonyms = set()
    antonyms = set()

    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())

    print(f"\nWord: '{word}'")
    print(f"  Synonyms : {sorted(synonyms)[:10]}")
    print(f"  Antonyms : {sorted(antonyms)[:10]}")

    # WordNet hierarchy
    synsets = wn.synsets(word)
    if synsets:
        s = synsets[0]
        print(f"  Definition: {s.definition()}")
        print(f"  Examples  : {s.examples()}")
        hypernyms = s.hypernyms()
        if hypernyms:
            print(f"  Hypernym  : {hypernyms[0].name()}")
