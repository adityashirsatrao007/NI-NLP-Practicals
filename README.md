# NLP Practicals 2026-27

Natural Language Processing lab practicals covering fundamental to advanced NLP concepts.

## Practicals

| # | Practical | File | Key Concepts |
|---|-----------|------|--------------|
| 1 | Tokenization & Word Frequency | `practical_1.py` | NLTK tokenizers, FreqDist |
| 2 | Synonym/Antonym using WordNet | `practical_2.py` | WordNet, lemmas, synsets |
| 3 | Bigram/Trigram Language Model & Regex | `practical_3.py` | N-grams, CFD, regex patterns |
| 4 | Lemmatization, Stemming & POS Tagging | `practical_4.py` | Porter/Lancaster stemmer, Penn Treebank tags |
| 5 | HMM POS Tagger & Chunker | `practical_5.py` | Hidden Markov Model, Viterbi, NP chunking |
| 6 | Named Entity Recognition | `practical_6.py` | NE chunking, entity types (PERSON, ORG, GPE) |
| 7 | Semantic Role Labelling | `practical_7.py` | Agent, Action, Patient, Location extraction |
| 8 | Text Classifier (Logistic Regression) | `practical_8.py` | TF-IDF, logistic regression, news classification |
| 9 | Movie Reviews Sentiment Classifier | `practical_9.py` | Sentiment analysis, movie reviews corpus |
| 10 | RNN for Sequence Labelling | `practical_10.py` | SimpleRNN, POS tagging, sequence-to-sequence |
| 11 | POS Tagging using LSTM | `practical_11.py` | Bidirectional LSTM, embedding layers |
| 12 | Word Sense Disambiguation (LSTM/GRU) | `practical_12.py` | WSD, LSTM vs GRU comparison |

## PDF Report

`NLP_Practicals_Report.pdf` — 24 pages, 2 pages per practical (code + output).

## Setup

```bash
git clone https://github.com/adityashirsatrao007/NLP-Practicals.git
cd NLP-Practicals
pip install -r requirements.txt
```

## Usage

```bash
# Run individual practicals
python experiments/practical_1.py

# Run all
for f in experiments/practical_*.py; do echo "=== $f ==="; python "$f"; echo; done
```

## Tech Stack

- **NLTK** — Tokenization, POS tagging, WordNet, NER, corpus access
- **scikit-learn** — TF-IDF, Logistic Regression, metrics
- **TensorFlow/Keras** — RNN, LSTM, GRU models
- **NumPy** — Data manipulation

## License

MIT
