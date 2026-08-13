"""
NLP Practicals Report - 24 pages exactly.
Odd pages: code. Even pages: output. No cover page.
"""
import os
from fpdf import FPDF

PROJECT = "/home/aditya/Documents/NLP-Practicals"
OUTPUT_PDF = os.path.join(PROJECT, "NLP_Practicals_Report.pdf")

practicals = [
    ("practical_1.py", "Tokenization & Word Frequency"),
    ("practical_2.py", "Synonyms & Antonyms using WordNet"),
    ("practical_3.py", "Bigram/Trigram Language Model & Regex"),
    ("practical_4.py", "Lemmatization, Stemming & POS Tagging"),
    ("practical_5.py", "HMM POS Tagger & Chunker"),
    ("practical_6.py", "Named Entity Recognition"),
    ("practical_7.py", "Semantic Role Labelling"),
    ("practical_8.py", "Text Classifier (Logistic Regression)"),
    ("practical_9.py", "Movie Reviews Sentiment Classifier"),
    ("practical_10.py", "RNN for Sequence Labelling"),
    ("practical_11.py", "POS Tagging using LSTM"),
    ("practical_12.py", "Word Sense Disambiguation (LSTM/GRU)"),
]


def sanitize(text):
    replacements = {
        "\u2500": "-", "\u2502": "|", "\u250c": "+", "\u2510": "+",
        "\u2514": "+", "\u2518": "+", "\u251c": "+", "\u2524": "+",
        "\u252c": "+", "\u2534": "+", "\u253c": "+", "\u2560": "=",
        "\u2550": "=", "\u2563": "=", "\u2561": "=", "\u2562": "=",
        "\u250f": "+", "\u2513": "+", "\u2517": "+", "\u251b": "+",
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
    skip = [
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
        "To enable the following",
    ]
    clean = []
    for line in raw.splitlines():
        if any(p in line for p in skip):
            continue
        clean.append(sanitize(line))
    while clean and clean[0].strip() == "":
        clean.pop(0)
    while clean and clean[-1].strip() == "":
        clean.pop()
    return "\n".join(clean)


def read_file(path):
    with open(path, "r") as f:
        return sanitize(f.read())


def get_output(practical_file):
    num = practical_file.replace(".py", "")
    path = os.path.join(PROJECT, "outputs", f"{num}_output.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            return clean_output(f.read())
    return "(output not captured)"


class Report(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=False)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.set_y(8)
        # Left: title
        self.set_x(self.l_margin)
        self.cell(95, 5, "NLP Practicals 2026-27", align="L")
        # Right: name
        self.set_x(self.l_margin + 95)
        self.cell(95, 5, "Aditya Shirsatrao", align="R")
        self.set_y(13)
        self.set_draw_color(160, 160, 160)
        self.set_line_width(0.2)
        self.line(self.l_margin, 14, self.w - self.r_margin, 14)
        self.set_y(16)

    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, f"Page {self.page_no()}/24", align="C")


def code_page(pdf, num, title, code):
    pdf.add_page()
    # Banner
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 60, 120)
    pdf.cell(0, 10, f"  Practical {num}  |  {title}", fill=True)
    pdf.ln(11)
    # Section label
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 6, "Source Code")
    pdf.ln(5)
    pdf.set_draw_color(30, 60, 120)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    # Code
    pdf.set_font("Courier", "", 6)
    pdf.set_text_color(30, 30, 30)
    lines = code.splitlines()
    start_y = pdf.get_y()
    available = 275 - start_y
    line_h = 2.6
    max_lines = int(available / line_h)
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1]
        lines.append("  # ... truncated, see full code in repo")
    pdf.set_fill_color(246, 246, 252)
    pdf.rect(10, start_y, 190, len(lines) * line_h + 2, "F")
    pdf.set_y(start_y + 1)
    for line in lines:
        if pdf.get_y() > 282:
            break
        pdf.cell(0, line_h, f"  {line[:115]}")
        pdf.ln(line_h)


def output_page(pdf, num, title, output):
    pdf.add_page()
    # Banner
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 60, 120)
    pdf.cell(0, 10, f"  Practical {num}  |  {title}  -  Output", fill=True)
    pdf.ln(11)
    # Section label
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 6, "Output")
    pdf.ln(5)
    pdf.set_draw_color(30, 60, 120)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    # Output text
    pdf.set_font("Courier", "", 6)
    pdf.set_text_color(30, 30, 30)
    lines = output.splitlines()
    start_y = pdf.get_y()
    available = 275 - start_y
    line_h = 2.6
    max_lines = int(available / line_h)
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1]
        lines.append("  ... (output truncated)")
    pdf.set_fill_color(250, 250, 245)
    pdf.rect(10, start_y, 190, len(lines) * line_h + 2, "F")
    pdf.set_y(start_y + 1)
    for line in lines:
        if pdf.get_y() > 282:
            break
        pdf.cell(0, line_h, f"  {line[:115]}")
        pdf.ln(line_h)


def main():
    pdf = Report()
    for i, (filename, title) in enumerate(practicals, 1):
        code = read_file(os.path.join(PROJECT, "experiments", filename))
        output = get_output(filename)
        code_page(pdf, i, title, code)
        output_page(pdf, i, title, output)
    pdf.output(OUTPUT_PDF)
    print(f"Generated: {OUTPUT_PDF}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
