"""
NLP Practicals Report Generator — Professional A4 PDF, 2 pages per experiment.
Page 1: Code (trimmed). Page 2: Output.
"""
import os
import subprocess
import re
from fpdf import FPDF

PROJECT = "/home/aditya/Documents/NLP-Practicals"
OUTPUT_PDF = os.path.join(PROJECT, "NLP_Practicals_Report.pdf")

experiments = [
    ("ex01_tokenization_freq.py", "Tokenization & Word Frequency"),
    ("ex02_wordnet.py", "Synonyms & Antonyms using WordNet"),
    ("ex03_ngram_regex.py", "Bigram/Trigram Language Model & Regex"),
    ("ex04_lemmatization_stemming.py", "Lemmatization, Stemming & POS Tagging"),
    ("ex05_hmm_chunker.py", "HMM POS Tagger & Chunker"),
    ("ex06_ner.py", "Named Entity Recognition"),
    ("ex07_srl.py", "Semantic Role Labelling"),
    ("ex08_text_classifier.py", "Text Classifier (Logistic Regression)"),
    ("ex09_sentiment.py", "Movie Reviews Sentiment Classifier"),
    ("ex10_rnn_seq.py", "RNN for Sequence Labelling"),
    ("ex11_lstm_pos.py", "POS Tagging using LSTM"),
    ("ex12_wsd.py", "Word Sense Disambiguation (LSTM/GRU)"),
]


def sanitize(text):
    """Replace non-latin-1 chars with ASCII equivalents."""
    replacements = {
        "\u2500": "-", "\u2502": "|", "\u250c": "+", "\u2510": "+",
        "\u2514": "+", "\u2518": "+", "\u251c": "+", "\u2524": "+",
        "\u252c": "+", "\u2534": "+", "\u253c": "+", "\u2560": "=",
        "\u2550": "=", "\u2563": "=", "\u2561": "=", "\u2562": "=",
        "\u2501": "-", "\u2503": "|", "\u250f": "+", "\u2513": "+",
        "\u2517": "+", "\u251b": "+", "\u2523": "+", "\u252f": "+",
        "\u2537": "+", "\u253f": "+",
        "\u2580": "=", "\u2584": "=", "\u2588": "=", "\u2591": ".",
        "\u2592": ":", "\u2593": "#",
    }
    result = []
    for ch in text:
        if ord(ch) < 128:
            result.append(ch)
        elif ch in replacements:
            result.append(replacements[ch])
        else:
            result.append("?")
    return "".join(result)


def clean_output(raw):
    lines = raw.splitlines()
    clean = []
    skip_patterns = [
        "nltk_data", "Security Violation", "urlopen", "pathsec",
        "NLTK_ALLOW", "for msg in", "UserWarning", "self.incr_download",
        "cuda_fft", "cuda_dnn", "cuda_blas", "computation_placer",
        "absl::", "gpu_device", "WARNING: All log", "Unable to register",
        "TensorFlow binary", "Loaded cuDNN", "MLIR crash",
        "Compiled cluster", "XLA service", "StreamExecutor",
        "service.cc", "device_compiler", "gpu_device.cc",
        "Created device", "MLIR_CRASH", "I0000", "W0000", "E0000",
        "UndefinedMetricWarning", "_warn_prf", "_classification.py",
        "Layer 'global_max", "Layer 'global_avg",
    ]
    for line in lines:
        if any(p in line for p in skip_patterns):
            continue
        clean.append(sanitize(line))
    while clean and clean[0].strip() == "":
        clean.pop(0)
    while clean and clean[-1].strip() == "":
        clean.pop()
    return "\n".join(clean)


def read_code(filepath):
    with open(filepath, "r") as f:
        return sanitize(f.read())


def get_output(experiment_file):
    # ex01_tokenization_freq.py -> ex01_output.txt
    num = experiment_file.split("_")[0]  # "ex01"
    output_file = os.path.join(PROJECT, "outputs", f"{num}_output.txt")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            return clean_output(f.read())
    return "(output not captured)"


class NLPReport(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=False)

    def header(self):
        if self.page_no() > 1:  # skip cover
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 5, "NLP Practicals 2026-27  |  Aditya Shirsatrao", align="C")
            self.ln(3)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")

    def cover_page(self):
        self.add_page()
        self.ln(60)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(30, 60, 120)
        self.cell(0, 15, "Natural Language Processing", align="C")
        self.ln(14)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(60, 60, 60)
        self.cell(0, 12, "Practical Lab Manual", align="C")
        self.ln(20)

        self.set_draw_color(30, 60, 120)
        self.set_line_width(0.8)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(15)

        self.set_font("Helvetica", "", 13)
        self.set_text_color(80, 80, 80)
        info_lines = [
            "Academic Year: 2026-27",
            "",
            "Name: Aditya Shirsatrao",
            "GitHub: adityashirsatrao007",
            "",
            "12 Experiments covering:",
            "Tokenization, WordNet, N-grams, Stemming,",
            "HMM, NER, SRL, Classification, Sentiment,",
            "RNN, LSTM, and Word Sense Disambiguation",
        ]
        for line in info_lines:
            self.cell(0, 8, line, align="C")
            self.ln(8)

    def exp_title_page(self, num, title):
        self.add_page()
        # Experiment number
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.set_fill_color(30, 60, 120)
        self.cell(0, 12, f"  Experiment {num:02d}", fill=True)
        self.ln(12)
        # Title
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 60, 120)
        self.ln(3)
        self.cell(0, 10, title)
        self.ln(12)

    def code_page(self, code, num):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 60, 120)
        self.cell(0, 7, "Source Code")
        self.ln(6)
        self.set_draw_color(30, 60, 120)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

        # Code block
        self.set_font("Courier", "", 6.5)
        self.set_text_color(40, 40, 40)
        self.set_fill_color(245, 245, 250)

        lines = code.splitlines()
        # Calculate available height
        start_y = self.get_y()
        available = 270 - start_y  # A4 height ~297, margins ~14 top+bottom, minus header
        line_h = 2.8
        max_lines = int(available / line_h)

        # Trim code to fit
        if len(lines) > max_lines:
            lines = lines[:max_lines - 2]
            lines.append("    # ... (truncated for page limit)")
            lines.append("    # See full code in GitHub repo")

        # Draw code background
        self.set_fill_color(245, 245, 250)
        self.rect(10, start_y, 190, len(lines) * line_h + 2, "F")

        self.set_y(start_y + 1)
        for line in lines:
            if self.get_y() > 280:
                break
            # Truncate long lines
            display = line[:110]
            self.cell(0, line_h, f"  {display}")
            self.ln(line_h)

    def output_page(self, output, num):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 60, 120)
        self.cell(0, 7, "Output")
        self.ln(6)
        self.set_draw_color(30, 60, 120)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

        # Output block
        self.set_font("Courier", "", 6.5)
        self.set_text_color(40, 40, 40)
        self.set_fill_color(250, 250, 245)

        lines = output.splitlines()
        start_y = self.get_y()
        available = 270 - start_y
        line_h = 2.8
        max_lines = int(available / line_h)

        if len(lines) > max_lines:
            lines = lines[:max_lines - 2]
            lines.append("  ... (output truncated)")
            lines.append(f"  Total output lines: {len(lines)}")

        self.set_fill_color(250, 250, 245)
        self.rect(10, start_y, 190, len(lines) * line_h + 2, "F")

        self.set_y(start_y + 1)
        for line in lines:
            if self.get_y() > 280:
                break
            display = line[:110]
            self.cell(0, line_h, f"  {display}")
            self.ln(line_h)


def main():
    pdf = NLPReport()
    pdf.cover_page()

    for i, (filename, title) in enumerate(experiments, 1):
        filepath = os.path.join(PROJECT, "experiments", filename)
        code = read_code(filepath)
        output = get_output(filename)

        # Page 1: Title + Code
        pdf.exp_title_page(i, title)
        pdf.code_page(code, i)

        # Page 2: Output
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(30, 60, 120)
        pdf.cell(0, 12, f"  Experiment {i:02d}  -  Output", fill=True)
        pdf.ln(14)
        pdf.output_page(output, i)

    pdf.output(OUTPUT_PDF)
    print(f"PDF generated: {OUTPUT_PDF}")
    print(f"Total pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
