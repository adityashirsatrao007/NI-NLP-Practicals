# NLP Practicals 2026-27

Natural Language Processing lab experiments covering fundamental to advanced NLP concepts.

## Experiments

| # | Experiment | File | Key Concepts |
|---|-----------|------|--------------|
| 1 | Tokenization & Word Frequency | `ex01_tokenization_freq.py` | NLTK tokenizers, FreqDist |
| 2 | Synonym/Antonym using WordNet | `ex02_wordnet.py` | WordNet, lemmas, synsets |
| 3 | Bigram/Trigram Language Model & Regex | `ex03_ngram_regex.py` | N-grams, CFD, regex patterns |
| 4 | Lemmatization, Stemming & POS Tagging | `ex04_lemmatization_stemming.py` | Porter/Lancaster stemmer, Penn Treebank tags |
| 5 | HMM POS Tagger & Chunker | `ex05_hmm_chunker.py` | Hidden Markov Model, Viterbi, NP chunking |
| 6 | Named Entity Recognition | `ex06_ner.py` | NE chunking, entity types (PERSON, ORG, GPE) |
| 7 | Semantic Role Labelling | `ex07_srl.py` | Agent, Action, Patient, Location extraction |
| 8 | Text Classifier (Logistic Regression) | `ex08_text_classifier.py` | TF-IDF, logistic regression, news classification |
| 9 | Movie Reviews Sentiment Classifier | `ex09_sentiment.py` | Sentiment analysis, movie reviews corpus |
| 10 | RNN for Sequence Labelling | `ex10_rnn_seq.py` | SimpleRNN, POS tagging, sequence-to-sequence |
| 11 | POS Tagging using LSTM | `ex11_lstm_pos.py` | Bidirectional LSTM, embedding layers |
| 12 | Word Sense Disambiguation (LSTM/GRU) | `ex12_wsd.py` | WSD, LSTM vs GRU comparison |

## Setup

```bash
# Clone the repo
git clone https://github.com/adityashirsatrao007/NLP-Practicals.git
cd NLP-Practicals

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (auto-downloaded by scripts, but manual option):
python -c "import nltk; nltk.download('all')"
```

## Usage

```bash
# Run individual experiments
python experiments/ex01_tokenization_freq.py
python experiments/ex02_wordnet.py
# ... etc

# Or run all
for f in experiments/ex*.py; do echo "=== $f ==="; python "$f"; echo; done
```

## Tech Stack

- **NLTK** — Tokenization, POS tagging, WordNet, NER, corpus access
- **spaCy** — Advanced NLP pipelines (available for extension)
- **scikit-learn** — TF-IDF, Logistic Regression, metrics
- **TensorFlow/Keras** — RNN, LSTM, GRU models
- **NumPy/Pandas** — Data manipulation

## License

MIT
