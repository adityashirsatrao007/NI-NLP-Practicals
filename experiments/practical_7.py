"""
Expt 7: Implement Semantic Role Labelling to identify named entities.
"""
import nltk
import re

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

print("=" * 60)
print("EXPERIMENT 7: Semantic Role Labelling (SRL)")
print("=" * 60)


def extract_srl(sentence):
    """Extract basic semantic roles using pattern matching and dependency heuristics."""
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    roles = {
        "AGENT": [],       # Who did it (subject)
        "ACTION": [],      # What happened (verb)
        "PATIENT": [],     # Who/what was affected (object)
        "INSTRUMENT": [],  # How/with what
        "LOCATION": [],    # Where
        "TIME": [],        # When
    }

    # Find verb (ACTION)
    verb_idx = None
    for i, (word, tag) in enumerate(tagged):
        if tag.startswith("VB") and tag != "VBG":
            roles["ACTION"].append(word)
            verb_idx = i
            break

    if verb_idx is None:
        return roles

    # Agent: NPs before verb (NNP, NN, PRP, DT+NN)
    for i in range(verb_idx - 1, -1, -1):
        word, tag = tagged[i]
        if tag in ("NNP", "NNPS", "PRP", "NN", "NNS"):
            roles["AGENT"].insert(0, word)
        elif tag == "DT":
            roles["AGENT"].insert(0, word)
            break
        elif tag in ("IN", "TO", "WDT", "WP"):
            break

    # Patient: NPs after verb
    for i in range(verb_idx + 1, len(tagged)):
        word, tag = tagged[i]
        if tag in ("NNP", "NNPS", "NN", "NNS", "PRP", "DT"):
            roles["PATIENT"].append(word)
        elif tag == "IN":
            # Check for location/time
            if i + 1 < len(tagged):
                next_word, next_tag = tagged[i + 1]
                if next_tag in ("NNP", "NNPS"):
                    roles["LOCATION"].append(next_word)
            break

    # Location markers
    loc_preps = {"in", "at", "on", "near", "from", "to", "into"}
    for i, (word, tag) in enumerate(tagged):
        if word.lower() in loc_preps and i + 1 < len(tagged):
            next_word, next_tag = tagged[i + 1]
            if next_tag in ("NNP", "NNPS"):
                if not roles["LOCATION"]:
                    roles["LOCATION"].append(next_word)

    # Time markers
    time_tags = {"CD"}
    for i, (word, tag) in enumerate(tagged):
        if tag == "CD" and i > verb_idx:
            roles["TIME"].append(word)

    return roles


sentences = [
    "Barack Obama signed the healthcare bill in Washington on Monday.",
    "The chef cooked a delicious meal in the kitchen.",
    "Google acquired YouTube for 1.65 billion dollars.",
    "Mary sent a letter to John from New York.",
    "The teacher explained the concept to students in the classroom.",
]

for i, sent in enumerate(sentences, 1):
    print(f"\n[{i}] {sent}")
    roles = extract_srl(sent)
    for role, values in roles.items():
        if values:
            print(f"  {role:12}: {' '.join(values)}")

# Named entities + SRL combined
print("\n--- Combined NER + SRL Analysis ---")
for i, sent in enumerate(sentences, 1):
    tokens = nltk.word_tokenize(sent)
    tagged = nltk.pos_tag(tokens)

    print(f"\n[{i}] {sent}")
    print(f"  POS Tags: {[(w, t) for w, t in tagged if t in ('NNP', 'NNPS', 'VB', 'VBD', 'VBZ')]}")

    roles = extract_srl(sent)
    print(f"  Semantic Roles:")
    for role, values in roles.items():
        if values:
            print(f"    {role:12} = {' '.join(values)}")
