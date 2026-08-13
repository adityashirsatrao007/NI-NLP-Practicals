"""
Expt 5: Implement HMM for POS tagging. Build a Chunker.
"""
import random

import nltk
from nltk import RegexpParser
from nltk.corpus import brown

nltk.download("brown", quiet=True)
nltk.download("nps_chat", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

print("=" * 60)
print("EXPERIMENT 5: HMM POS Tagger & Chunker")
print("=" * 60)

brown_tagged = brown.tagged_sents(categories="news", tagset="universal")

# Split train/test
split = int(len(brown_tagged) * 0.8)
train_sents = brown_tagged[:split]
test_sents = brown_tagged[split:]


class HMMTagger:
    def __init__(self):
        self.tag_freq = {}
        self.transition = {}
        self.emission = {}

    def train(self, tagged_sents):
        for sent in tagged_sents:
            prev_tag = "<START>"
            for word, tag in sent:
                # Transition counts
                self.transition.setdefault(prev_tag, {})
                self.transition[prev_tag][tag] = self.transition[prev_tag].get(tag, 0) + 1
                # Emission counts
                self.emission.setdefault(tag, {})
                self.emission[tag][word] = self.emission[tag].get(word, 0) + 1
                # Tag counts
                self.tag_freq[tag] = self.tag_freq.get(tag, 0) + 1
                prev_tag = tag
            self.transition.setdefault(prev_tag, {})
            self.transition[prev_tag]["<END>"] = self.transition[prev_tag].get("<END>", 0) + 1

    def _transition_prob(self, prev_tag, tag):
        if prev_tag not in self.transition:
            return 1e-6
        total = sum(self.transition[prev_tag].values())
        return self.transition[prev_tag].get(tag, 0) / total

    def _emission_prob(self, tag, word):
        if tag not in self.emission:
            return 1e-6
        total = sum(self.emission[tag].values())
        return self.emission[tag].get(word, 1e-6) / total

    def _viterbi(self, words):
        tags = list(self.tag_freq.keys())
        n = len(words)
        if n == 0:
            return []

        # Viterbi table
        viterbi = [{}]
        backptr = [{}]

        for tag in tags:
            viterbi[0][tag] = self._transition_prob("<START>", tag) * self._emission_prob(tag, words[0])
            backptr[0][tag] = "<START>"

        for t in range(1, n):
            viterbi.append({})
            backptr.append({})
            for tag in tags:
                max_prob = -1
                best_prev = tags[0]
                for prev_tag in tags:
                    prob = (viterbi[t-1][prev_tag]
                            * self._transition_prob(prev_tag, tag)
                            * self._emission_prob(tag, words[t]))
                    if prob > max_prob:
                        max_prob = prob
                        best_prev = prev_tag
                viterbi[t][tag] = max_prob
                backptr[t][tag] = best_prev

        # Backtrack
        best_last = max(tags, key=lambda t: viterbi[n-1][t])
        best_path = [best_last]
        for t in range(n-2, -1, -1):
            best_path.insert(0, backptr[t+1][best_path[0]])
        return best_path

    def tag(self, words):
        return list(zip(words, self._viterbi(words)))

    def accuracy(self, test_sents):
        correct = total = 0
        for sent in test_sents:
            words = [w for w, t in sent]
            gold_tags = [t for w, t in sent]
            pred_tags = [t for w, t in self.tag(words)]
            correct += sum(1 for g, p in zip(gold_tags, pred_tags) if g == p)
            total += len(gold_tags)
        return correct / total


print("\n--- Training HMM POS Tagger (Brown corpus, news) ---")
hmm = HMMTagger()
hmm.train(train_sents)
acc = hmm.accuracy(test_sents)
print(f"HMM Tagger Accuracy: {acc:.2%}")

# Test on custom sentence
test_sentence = "The quick brown fox jumps over the lazy dog"
tokens = test_sentence.split()
tagged = hmm.tag(tokens)
print(f"\nCustom: '{test_sentence}'")
print(f"Tagged: {tagged}")

# ── Chunker ──
print("\n--- NP Chunker (RegexpParser) ---")
grammar = r"""
    NP: {<DT>?<JJ>*<NN.*>+}
    VP: {<VB.*><NP|PP>?}
    PP: {<IN><NP>}
"""
chunker = RegexpParser(grammar)

test_chunk = nltk.pos_tag(nltk.word_tokenize("The quick brown fox jumps over the lazy dog"))
tree = chunker.parse(test_chunk)
print(f"Chunks found:")
for subtree in tree.subtrees():
    if subtree.label() != "S":
        print(f"  {subtree.label()}: {' '.join(w for w, t in subtree.leaves())}")

print("\nTree saved to outputs/ex05_chunk_tree.txt")
with open("outputs/ex05_chunk_tree.txt", "w") as f:
    f.write(str(tree))
