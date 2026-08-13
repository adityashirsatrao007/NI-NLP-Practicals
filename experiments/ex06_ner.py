"""
Expt 6: Implement Named Entity Recognizer.
"""
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from collections import Counter

nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("maxent_ne_chunker_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("words", quiet=True)

print("=" * 60)
print("EXPERIMENT 6: Named Entity Recognition (NER)")
print("=" * 60)

sentences = [
    "Barack Obama was born in Hawaii and served as the 44th President of the United States.",
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    "Google was created by Larry Page and Sergey Brin at Stanford University.",
    "Elon Musk founded SpaceX in Hawthorne and Tesla in Austin, Texas.",
    "Satya Nadella is the CEO of Microsoft in Redmond, Washington.",
]

entity_types = Counter()

for i, sent in enumerate(sentences, 1):
    print(f"\n[{i}] {sent}")
    tokens = word_tokenize(sent)
    tagged = pos_tag(tokens)
    tree = ne_chunk(tagged)

    entities = []
    for subtree in tree:
        if hasattr(subtree, "label"):
            entity_name = " ".join(word for word, tag in subtree.leaves())
            entity_type = subtree.label()
            entities.append((entity_name, entity_type))
            entity_types[entity_type] += 1

    if entities:
        print(f"  Entities found:")
        for name, etype in entities:
            print(f"    {etype}: {name}")
    else:
        print("  No entities found.")

print(f"\n--- Entity Type Distribution ---")
for etype, count in entity_types.most_common():
    print(f"  {etype}: {count}")

# Extract structured data
print("\n--- Structured Extraction ---")
for i, sent in enumerate(sentences, 1):
    tokens = word_tokenize(sent)
    tagged = pos_tag(tokens)
    tree = ne_chunk(tagged)

    people = []
    organizations = []
    locations = []

    for subtree in tree:
        if hasattr(subtree, "label"):
            name = " ".join(w for w, t in subtree.leaves())
            if subtree.label() == "PERSON":
                people.append(name)
            elif subtree.label() == "ORGANIZATION":
                organizations.append(name)
            elif subtree.label() == "GPE":
                locations.append(name)

    print(f"\n  Sentence {i}:")
    print(f"    People      : {people or 'None'}")
    print(f"    Organizations: {organizations or 'None'}")
    print(f"    Locations    : {locations or 'None'}")
