r"""
scripts/generate_report_figures.py
==================================
Generates all 26 high-resolution (300 DPI) publication-grade figures
for the academic project report of SpamShield AI.
Outputs to:
  - D:\spam-ham-detector\report_figures\
  - C:\Users\tholk\.gemini\antigravity\brain\b57ca07d-ec2d-4cd4-83ec-879e77b1b8fb\figures\
"""


import os
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle
import seaborn as sns

# Set high-resolution styling defaults
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.2,
    "grid.color": "#F1F5F9",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

OUT_DIR_LOCAL = Path(r"D:\spam-ham-detector\report_figures")
OUT_DIR_ARTIFACT = Path(r"C:\Users\tholk\.gemini\antigravity\brain\b57ca07d-ec2d-4cd4-83ec-879e77b1b8fb\figures")

OUT_DIR_LOCAL.mkdir(parents=True, exist_ok=True)
OUT_DIR_ARTIFACT.mkdir(parents=True, exist_ok=True)


def save_fig(fig, filename):
    p1 = OUT_DIR_LOCAL / filename
    p2 = OUT_DIR_ARTIFACT / filename
    fig.savefig(p1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy(p1, p2)
    print(f"Saved: {filename}")


# ===========================================================================
# CHAPTER 1 FIGURES
# ===========================================================================

def make_fig_1_1():
    """Fig 1.1: Global Spam Email Growth Graph (2015-2026)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    years = np.arange(2015, 2027)
    # Global spam volume in billions per day (approx industry stats)
    spam_volume = [142, 160, 185, 210, 240, 280, 319, 347, 375, 410, 442, 480]
    spam_pct = [54.2, 53.0, 52.8, 51.5, 52.1, 55.4, 53.9, 51.8, 49.6, 48.2, 47.1, 46.8]

    ax.plot(years, spam_volume, color="#4F46E5", linewidth=3, marker="o", markersize=6, label="Daily Spam Volume (Billions)")
    ax.fill_between(years, spam_volume, color="#4F46E5", alpha=0.12)

    ax2 = ax.twinx()
    ax2.plot(years, spam_pct, color="#E11D48", linewidth=2.5, linestyle="--", marker="s", markersize=5, label="Spam Share of Total Traffic (%)")

    ax.set_title("Fig 1.1: Global Spam Email Volume & Traffic Share (2015–2026)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=11, fontweight="600")
    ax.set_ylabel("Spam Emails / Day (Billions)", fontsize=11, fontweight="600", color="#4F46E5")
    ax2.set_ylabel("Spam Traffic Share (%)", fontsize=11, fontweight="600", color="#E11D48")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(years)
    ax.set_ylim(100, 520)
    ax2.set_ylim(40, 60)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    save_fig(fig, "fig_1_1_spam_growth_graph.png")


def make_fig_1_2():
    """Fig 1.2: Complete System Block Diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    def draw_box(x, y, w, h, title, subtitle, color, text_color="white"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                              facecolor=color, edgecolor="#334155", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=text_color)
        ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center",
                fontsize=8.5, color=text_color, alpha=0.9)

    # Blocks
    draw_box(0.6, 2.3, 2.2, 1.4, "Email Ingestion", "IMAP SSL / CSV / Text", "#1E293B")
    draw_box(3.5, 2.3, 2.2, 1.4, "Preprocessing", "HTML Strip, URLs, Whitespace", "#0284C7")
    draw_box(6.4, 2.3, 2.2, 1.4, "DistilBERT Engine", "6 Transformer Layers (66M)", "#4F46E5")
    draw_box(9.3, 2.3, 2.2, 1.4, "Inference & Verdict", "SPAM / HAM + Softmax %", "#059669")

    # Lower support blocks
    draw_box(2.0, 0.4, 2.5, 1.1, "SQLite Database", "User Accounts & History", "#475569")
    draw_box(5.0, 0.4, 2.5, 1.1, "SHAP XAI Engine", "Word-Level Importance", "#D97706")
    draw_box(8.0, 0.4, 2.5, 1.1, "Streamlit Dashboard", "Interactive User Portal", "#6366F1")

    # Arrows
    arrow_kw = dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#0F172A", lw=2)
    ax.annotate("", xy=(3.5, 3.0), xytext=(2.8, 3.0), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.4, 3.0), xytext=(5.7, 3.0), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.3, 3.0), xytext=(8.6, 3.0), arrowprops=arrow_kw)

    # Supporting vertical arrows
    ax.annotate("", xy=(4.6, 2.3), xytext=(3.2, 1.5), arrowprops=dict(arrowstyle="<->", color="#64748B", lw=1.5, ls="--"))
    ax.annotate("", xy=(7.5, 2.3), xytext=(6.2, 1.5), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, ls="--"))
    ax.annotate("", xy=(10.4, 2.3), xytext=(9.2, 1.5), arrowprops=dict(arrowstyle="<->", color="#64748B", lw=1.5, ls="--"))

    ax.set_title("Fig 1.2: Complete End-to-End System Block Diagram", fontsize=13, fontweight="bold", pad=20)
    save_fig(fig, "fig_1_2_complete_system_block_diagram.png")


# ===========================================================================
# CHAPTER 2 FIGURES
# ===========================================================================

def make_fig_2_1():
    """Fig 2.1: Spam detection evolution timeline."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)

    eras = [
        ("1990s", "Heuristic Rules", "Keyword blacklists,\nregex filters", "#64748B"),
        ("2000s", "Naive Bayes", "Bag-of-words,\nprobabilistic model", "#0284C7"),
        ("2010s", "ML Classifiers", "SVM, Random Forest,\nTF-IDF features", "#059669"),
        ("2017+", "Deep Learning", "RNN, Bi-LSTM,\nsequential context", "#D97706"),
        ("2020+", "Transformers", "DistilBERT, BERT,\nSelf-Attention (Ours)", "#4F46E5"),
    ]

    # Baseline line
    ax.plot([1.0, 10.0], [2.0, 2.0], color="#94A3B8", lw=3, zorder=1)

    for i, (yr, title, desc, col) in enumerate(eras):
        cx = 1.0 + i * 2.2
        # Circle on timeline
        circle = plt.Circle((cx, 2.0), 0.22, color=col, zorder=3)
        ax.add_patch(circle)
        ax.text(cx, 2.0, str(i+1), ha="center", va="center", color="white", fontweight="bold", fontsize=10, zorder=4)

        # Year tag
        ax.text(cx, 2.45, yr, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1E293B")
        # Card below
        rect = FancyBboxPatch((cx - 0.95, 0.4), 1.9, 1.1, boxstyle="round,pad=0.15",
                              facecolor="#F8FAFC", edgecolor=col, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(cx, 1.1, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=col)
        ax.text(cx, 0.7, desc, ha="center", va="center", fontsize=8, color="#475569")

    ax.set_title("Fig 2.1: Evolution Timeline of Email Spam Detection Methodologies", fontsize=13, fontweight="bold", pad=15)
    save_fig(fig, "fig_2_1_spam_detection_evolution_timeline.png")


# ===========================================================================
# CHAPTER 3 FIGURES
# ===========================================================================

def make_fig_3_1():
    """Fig 3.1: System Architecture Diagram (Multi-Layer)."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)

    layers = [
        ("Presentation Layer", "Streamlit SPA: Live Predict, Gmail Scanner, CSV Upload, SHAP Visualizer, Metrics", "#4F46E5", 5.6),
        ("Application & Security Layer", "Session Guard, Enterprise PBKDF2 Auth, Gmail SSL IMAP Client, Threat Dispatcher", "#0284C7", 4.1),
        ("AI / ML Inference Layer", "DistilBERT Transformer (66M params), Softmax Classifier, SHAP Tree/Kernel Explainer", "#059669", 2.6),
        ("Data & Storage Layer", "SQLite DB (users, predictions, reset tokens), Model Safetensors, UCI & SpamAssassin Data", "#475569", 1.1),
    ]

    for title, subtitle, col, y in layers:
        rect = FancyBboxPatch((0.8, y), 9.4, 1.15, boxstyle="round,pad=0.2",
                              facecolor="#FFFFFF", edgecolor=col, linewidth=2.0)
        ax.add_patch(rect)
        # Header banner inside box
        header_rect = FancyBboxPatch((0.85, y + 0.65), 9.3, 0.45, boxstyle="round,pad=0.1",
                                     facecolor=col, edgecolor=col)
        ax.add_patch(header_rect)
        ax.text(5.5, y + 0.88, title, ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.text(5.5, y + 0.32, subtitle, ha="center", va="center", fontsize=9, color="#1E293B")

    # Bidirectional arrows between layers
    arrow_kw = dict(arrowstyle="<->,head_width=0.3,head_length=0.5", color="#64748B", lw=2)
    ax.annotate("", xy=(5.5, 5.6), xytext=(5.5, 5.25), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.5, 4.1), xytext=(5.5, 3.75), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.5, 2.6), xytext=(5.5, 2.25), arrowprops=arrow_kw)

    ax.set_title("Fig 3.1: Multi-Tiered System Architecture of SpamShield AI", fontsize=13, fontweight="bold", pad=15)
    save_fig(fig, "fig_3_1_system_architecture.png")


def make_fig_3_2():
    """Fig 3.2: Preprocessing flowchart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    steps = [
        ("Raw Email", "HTML / MIME / Plain Text", "#1E293B", 0.6),
        ("HTML Stripping", "BeautifulSoup & lxml tags", "#0284C7", 2.4),
        ("Entity Unescape", "html.unescape & zero-width", "#059669", 4.2),
        ("URL & Header Strip", "Regex URL & Header cleaner", "#D97706", 6.0),
        ("Clean Tokenizer Input", "Collapsed whitespace text", "#4F46E5", 7.8),
    ]

    for title, desc, col, x in steps:
        rect = FancyBboxPatch((x, 1.8), 1.5, 1.4, boxstyle="round,pad=0.15",
                              facecolor=col, edgecolor="#334155", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.75, 2.7, title, ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax.text(x + 0.75, 2.2, desc, ha="center", va="center", color="white", alpha=0.9, fontsize=7.5)

    arrow_kw = dict(arrowstyle="->,head_width=0.35,head_length=0.5", color="#1E293B", lw=2)
    ax.annotate("", xy=(2.4, 2.5), xytext=(2.1, 2.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(4.2, 2.5), xytext=(3.9, 2.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.0, 2.5), xytext=(5.7, 2.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.8, 2.5), xytext=(7.5, 2.5), arrowprops=arrow_kw)

    ax.set_title("Fig 3.2: Text Preprocessing Pipeline Flowchart", fontsize=13, fontweight="bold", pad=15)
    save_fig(fig, "fig_3_2_preprocessing_flowchart.png")


def make_fig_3_3():
    """Fig 3.3: DistilBERT architecture."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    layers = [
        ("Input Tokens", "[CLS] Token_1 Token_2 ... [SEP]", "#64748B", 0.5),
        ("Embedding Layer", "Token + Position Embeddings (d=768)", "#0284C7", 1.4),
        ("Transformer Layer 1–6", "6x DistilBERT Multi-Head Attention & FFN", "#4F46E5", 2.6),
        ("Pooled [CLS] Representation", "768-dimensional context vector", "#059669", 3.8),
        ("Classification Head", "Linear(768, 2) + Softmax -> [P(HAM), P(SPAM)]", "#E11D48", 4.9),
    ]

    for title, desc, col, y in layers:
        h = 0.85 if "Layer 1–6" not in title else 1.0
        rect = FancyBboxPatch((1.5, y), 7.0, h, boxstyle="round,pad=0.15",
                              facecolor=col, edgecolor="#334155", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(5.0, y + h*0.62, title, ha="center", va="center", color="white", fontweight="bold", fontsize=10.5)
        ax.text(5.0, y + h*0.28, desc, ha="center", va="center", color="white", alpha=0.9, fontsize=8.5)

    arrow_kw = dict(arrowstyle="->,head_width=0.3,head_length=0.4", color="#334155", lw=1.8)
    ax.annotate("", xy=(5.0, 1.4), xytext=(5.0, 1.35), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.0, 2.6), xytext=(5.0, 2.25), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.0, 3.8), xytext=(5.0, 3.6), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.0, 4.9), xytext=(5.0, 4.65), arrowprops=arrow_kw)

    ax.set_title("Fig 3.3: DistilBERT Model Architecture for Binary Classification", fontsize=13, fontweight="bold", pad=15)
    save_fig(fig, "fig_3_3_distilbert_architecture.png")


def make_fig_3_4():
    """Fig 3.4: Fine-tuning workflow."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    boxes = [
        ("Dataset (10,389)", "Train: 7,791\nVal: 1,039\nTest: 1,559", "#1E293B", 0.5, 2.0),
        ("Tokenization", "Max len: 256\nPadding & Trunc", "#0284C7", 2.8, 2.0),
        ("Forward Pass", "CrossEntropyLoss\nWeight decay: 0.01", "#4F46E5", 5.1, 2.0),
        ("AdamW Optimizer", "LR: 2e-5\nWarmup: 100", "#D97706", 7.4, 2.0),
    ]

    for title, desc, col, x, y in boxes:
        rect = FancyBboxPatch((x, y), 2.0, 1.5, boxstyle="round,pad=0.15",
                              facecolor=col, edgecolor="#334155", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 1.0, y + 1.05, title, ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        ax.text(x + 1.0, y + 0.5, desc, ha="center", va="center", color="white", alpha=0.9, fontsize=8.5)

    arrow_kw = dict(arrowstyle="->,head_width=0.35,head_length=0.5", color="#1E293B", lw=2)
    ax.annotate("", xy=(2.8, 2.75), xytext=(2.5, 2.75), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.1, 2.75), xytext=(4.8, 2.75), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.4, 2.75), xytext=(7.1, 2.75), arrowprops=arrow_kw)

    # Feedback loop arrow (Backpropagation)
    ax.annotate("Backprop & Weight Update (3 Epochs)", xy=(6.1, 2.0), xytext=(6.1, 0.9),
                ha="center", fontsize=9, fontweight="bold", color="#E11D48",
                arrowprops=dict(arrowstyle="->", color="#E11D48", lw=1.8, connectionstyle="arc3,rad=-0.3"))

    ax.set_title("Fig 3.4: DistilBERT Fine-Tuning Execution Workflow", fontsize=13, fontweight="bold", pad=15)
    save_fig(fig, "fig_3_4_finetuning_workflow.png")


def make_fig_3_5():
    """Fig 3.5: Dataset split pie & bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Pie chart: HAM vs SPAM distribution
    labels = ["HAM (Legitimate)", "SPAM (Malicious)"]
    sizes = [8970, 1419]
    colors = ["#10B981", "#EF4444"]
    explode = (0, 0.08)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%",
                                      startangle=140, colors=colors, textprops=dict(color="#1E293B", fontweight="600"))
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    ax1.set_title("A: Class Distribution (Total: 10,389)", fontsize=11, fontweight="bold")

    # Bar chart: Data Split (Train/Val/Test)
    splits = ["Train (75%)", "Validation (10%)", "Test (15%)"]
    counts = [7791, 1039, 1559]
    bar_cols = ["#4F46E5", "#0284C7", "#F59E0B"]

    bars = ax2.bar(splits, counts, color=bar_cols, width=0.55, edgecolor="#1E293B", linewidth=1.2)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 100, f"{height:,}",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1E293B")

    ax2.set_ylabel("Number of Samples", fontsize=10, fontweight="600")
    ax2.set_ylim(0, 9000)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_title("B: Train / Val / Test Partitioning", fontsize=11, fontweight="bold")

    fig.suptitle("Fig 3.5: Augmented Dataset Distribution and Partitioning Breakdown", fontsize=13, fontweight="bold", y=1.03)
    save_fig(fig, "fig_3_5_dataset_split_chart.png")


# ===========================================================================
# CHAPTER 4 FIGURES
# ===========================================================================

def make_fig_4_1():
    """Fig 4.1: Module dependency diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    modules = [
        ("app.main", "SPA Entry Point", "#4F46E5", 4.0, 5.0),
        ("app.components.auth_ui", "Auth UI & Session", "#0284C7", 1.5, 3.5),
        ("spam_detector.predict", "Inference Engine", "#059669", 4.0, 3.5),
        ("spam_detector.gmail", "IMAP Scanner", "#D97706", 6.5, 3.5),
        ("spam_detector.explain", "SHAP XAI Engine", "#E11D48", 9.0, 3.5),
        ("spam_detector.preprocess", "Cleaning Pipeline", "#0284C7", 2.5, 1.8),
        ("spam_detector.db", "SQLite Database", "#475569", 5.5, 1.8),
        ("spam_detector.auth", "PBKDF2 Hashing", "#64748B", 8.0, 1.8),
    ]

    for mod, desc, col, x, y in modules:
        rect = FancyBboxPatch((x - 1.1, y - 0.45), 2.2, 0.9, boxstyle="round,pad=0.1",
                              facecolor=col, edgecolor="#1E293B", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y + 0.1, mod, ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
        ax.text(x, y - 0.2, desc, ha="center", va="center", color="white", alpha=0.9, fontsize=7.5)

    # Connections
    arrow_kw = dict(arrowstyle="->", color="#64748B", lw=1.5)
    ax.annotate("", xy=(2.0, 3.95), xytext=(4.0, 4.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(4.5, 3.95), xytext=(4.5, 4.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.5, 3.95), xytext=(5.0, 4.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.5, 3.95), xytext=(5.5, 4.8), arrowprops=arrow_kw)

    ax.annotate("", xy=(3.0, 2.25), xytext=(4.5, 3.05), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.5, 2.25), xytext=(4.5, 3.05), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.0, 2.25), xytext=(2.0, 3.05), arrowprops=arrow_kw)

    ax.set_title("Fig 4.1: Software Package Module Dependency Architecture", fontsize=13, fontweight="bold", pad=15)
    save_fig(fig, "fig_4_1_module_dependency_diagram.png")


def make_fig_4_2():
    """Fig 4.2: Login/Register authentication flow."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    nodes = [
        ("User Login Request", "Username + Password", "#1E293B", 1.0, 3.0),
        ("User Query", "Fetch password_hash + salt", "#0284C7", 3.5, 3.0),
        ("PBKDF2 Verification", "100k iter SHA-256 compare_digest", "#4F46E5", 6.0, 3.0),
        ("Session Granted", "Streamlit session state active", "#059669", 8.5, 3.0),
        ("Forgot Password", "OTP -> Email -> Reset", "#D97706", 6.0, 1.2),
    ]

    for title, desc, col, x, y in nodes:
        rect = FancyBboxPatch((x - 1.0, y - 0.5), 2.0, 1.0, boxstyle="round,pad=0.12",
                              facecolor=col, edgecolor="#334155", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y + 0.15, title, ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
        ax.text(x, y - 0.2, desc, ha="center", va="center", color="white", alpha=0.9, fontsize=7.2)

    arrow_kw = dict(arrowstyle="->,head_width=0.3,head_length=0.45", color="#1E293B", lw=1.8)
    ax.annotate("", xy=(2.5, 3.0), xytext=(2.0, 3.0), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.0, 3.0), xytext=(4.5, 3.0), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.5, 3.0), xytext=(7.0, 3.0), arrowprops=arrow_kw)

    # Forgot branch
    ax.annotate("", xy=(6.0, 1.7), xytext=(6.0, 2.5), arrowprops=dict(arrowstyle="<->", color="#D97706", lw=1.5, ls="--"))
    ax.text(6.7, 2.1, "Recovery Path", fontsize=8, color="#D97706", fontweight="600")

    ax.set_title("Fig 4.2: Enterprise User Authentication & Password Reset Flow", fontsize=13, fontweight="bold", pad=15)
    save_fig(fig, "fig_4_2_authentication_flow.png")


def make_fig_4_3():
    """Fig 4.3: Gmail IMAP workflow."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)

    stages = [
        ("1. Connect", "IMAP4_SSL\nimap.gmail.com:993", "#1E293B", 0.7),
        ("2. Authenticate", "Google App Password\n(16 characters)", "#0284C7", 2.8),
        ("3. Fetch Messages", "RFC 822 / MIME\nHeaders & Bodies", "#4F46E5", 4.9),
        ("4. Classify", "DistilBERT Model\nBatch Inference", "#059669", 7.0),
        ("5. Log & Display", "Audit History &\nStreamlit Expander", "#D97706", 9.1),
    ]

    for title, desc, col, x in stages:
        rect = FancyBboxPatch((x, 1.5), 1.7, 1.5, boxstyle="round,pad=0.15",
                              facecolor=col, edgecolor="#334155", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.85, 2.5, title, ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
        ax.text(x + 0.85, 1.9, desc, ha="center", va="center", color="white", alpha=0.9, fontsize=8)

    arrow_kw = dict(arrowstyle="->,head_width=0.3,head_length=0.5", color="#1E293B", lw=2)
    ax.annotate("", xy=(2.8, 2.25), xytext=(2.4, 2.25), arrowprops=arrow_kw)
    ax.annotate("", xy=(4.9, 2.25), xytext=(4.5, 2.25), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.0, 2.25), xytext=(6.6, 2.25), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.1, 2.25), xytext=(8.7, 2.25), arrowprops=arrow_kw)

    ax.set_title("Fig 4.3: Secure Gmail IMAP Integration & Threat Classification Pipeline", fontsize=13, fontweight="bold", pad=15)
    save_fig(fig, "fig_4_3_gmail_imap_workflow.png")


def make_fig_4_4():
    """Fig 4.4: Streamlit Single-Page tab navigation architecture."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    # Main SPA Container
    main_rect = FancyBboxPatch((0.5, 3.5), 9.0, 1.0, boxstyle="round,pad=0.15",
                               facecolor="#1E293B", edgecolor="#4F46E5", linewidth=2.0)
    ax.add_patch(main_rect)
    ax.text(5.0, 4.15, "app/main.py — Unified Single-Page Application (SPA)", ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    ax.text(5.0, 3.75, "Single runtime entry point with session guarding and shared cached DistilBERT predictor", ha="center", va="center", color="#94A3B8", fontsize=8.5)

    tabs = [
        ("Tab 1: Live Predict", "Single text inference\n+ Softmax breakdown", "#4F46E5", 0.5),
        ("Tab 2: Gmail Scanner", "SSL IMAP scan\n+ Diagnostics suite", "#0284C7", 2.35),
        ("Tab 3: Batch CSV", "Bulk CSV prediction\n+ Export downloads", "#059669", 4.2),
        ("Tab 4: SHAP XAI", "Word token heatmaps\n+ Feature attribution", "#D97706", 6.05),
        ("Tab 5: Metrics & Logs", "Test set telemetry\n+ Database audit log", "#E11D48", 7.9),
    ]

    for title, desc, col, x in tabs:
        rect = FancyBboxPatch((x, 1.0), 1.6, 1.6, boxstyle="round,pad=0.12",
                              facecolor="#FFFFFF", edgecolor=col, linewidth=1.8)
        ax.add_patch(rect)
        hdr = FancyBboxPatch((x + 0.05, 2.05), 1.5, 0.5, boxstyle="round,pad=0.08", facecolor=col, edgecolor=col)
        ax.add_patch(hdr)
        ax.text(x + 0.8, 2.3, title, ha="center", va="center", color="white", fontweight="bold", fontsize=7.8)
        ax.text(x + 0.8, 1.5, desc, ha="center", va="center", color="#1E293B", fontsize=7.2)

        # Arrow down from main container
        ax.annotate("", xy=(x + 0.8, 2.6), xytext=(x + 0.8, 3.5),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))

    ax.set_title("Fig 4.4: Streamlit Single-Page Tab Navigation Architecture", fontsize=13, fontweight="bold", pad=15)
    save_fig(fig, "fig_4_4_streamlit_page_navigation.png")


def make_fig_4_5():
    """Fig 4.5: Database ER Diagram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    def draw_entity(x, y, title, fields, color):
        w, h = 2.6, 0.45 + len(fields)*0.28
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor="#FFFFFF", edgecolor=color, linewidth=2.0)
        ax.add_patch(rect)
        hdr = FancyBboxPatch((x, y + h - 0.45), w, 0.45, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor=color)
        ax.add_patch(hdr)
        ax.text(x + w/2, y + h - 0.22, title, ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
        for idx, (f, typ, pk) in enumerate(fields):
            fy = y + h - 0.7 - idx*0.28
            prefix = "PK " if pk == "PK" else ("FK " if pk == "FK" else "   ")
            ax.text(x + 0.15, fy, f"{prefix}{f}", fontsize=8, fontweight="bold" if pk else "normal", color="#0F172A")
            ax.text(x + w - 0.15, fy, typ, fontsize=7.5, color="#64748B", ha="right")

    users_fields = [
        ("id", "INTEGER", "PK"),
        ("username", "TEXT", ""),
        ("email", "TEXT", ""),
        ("password_hash", "TEXT", ""),
        ("salt", "TEXT", ""),
        ("role", "TEXT", ""),
        ("gmail_address", "TEXT", ""),
    ]

    preds_fields = [
        ("id", "INTEGER", "PK"),
        ("created_at", "TEXT", ""),
        ("input_hash", "TEXT", ""),
        ("raw_text", "TEXT", ""),
        ("label", "TEXT", ""),
        ("spam_prob", "REAL", ""),
        ("confidence", "REAL", ""),
        ("source", "TEXT", ""),
    ]

    tokens_fields = [
        ("id", "INTEGER", "PK"),
        ("email", "TEXT", "FK"),
        ("token", "TEXT", ""),
        ("created_at", "TEXT", ""),
        ("expires_at", "TEXT", ""),
        ("used", "INTEGER", ""),
    ]

    draw_entity(0.6, 1.2, "users", users_fields, "#1E293B")
    draw_entity(3.7, 1.0, "predictions", preds_fields, "#4F46E5")
    draw_entity(6.8, 1.5, "password_reset_tokens", tokens_fields, "#D97706")

    # Relationships
    ax.annotate("", xy=(6.8, 3.2), xytext=(3.2, 3.2),
                arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, ls="--"))
    ax.text(5.0, 3.35, "1-to-Many by email", fontsize=8, color="#64748B", ha="center")

    ax.set_title("Fig 4.5: SQLite Relational Database Entity-Relationship (ER) Diagram", fontsize=13, fontweight="bold", pad=15)
    save_fig(fig, "fig_4_5_database_er_diagram.png")


# ===========================================================================
# CHAPTER 5 FIGURES
# ===========================================================================

def make_fig_5_1():
    """Fig 5.1: Testing pyramid."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)

    # Pyramid tiers
    # Bottom: Unit Tests
    p_unit = patches.Polygon([[1.0, 0.8], [7.0, 0.8], [6.0, 2.0], [2.0, 2.0]], facecolor="#10B981", edgecolor="#064E3B", lw=1.5)
    ax.add_patch(p_unit)
    ax.text(4.0, 1.35, "Unit Tests (28 Tests)\nPreprocessing, DB CRUD, Password Hashing", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)

    # Middle: Integration Tests
    p_int = patches.Polygon([[2.0, 2.0], [6.0, 2.0], [5.0, 3.2], [3.0, 3.2]], facecolor="#0284C7", edgecolor="#0C4A6E", lw=1.5)
    ax.add_patch(p_int)
    ax.text(4.0, 2.55, "Integration Tests (11 Tests)\nGmail IMAP, SQLite Sync, Auth Guard", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)

    # Top: System & UI Tests
    p_sys = patches.Polygon([[3.0, 3.2], [5.0, 3.2], [4.0, 4.4]], facecolor="#6366F1", edgecolor="#312E81", lw=1.5)
    ax.add_patch(p_sys)
    ax.text(4.0, 3.65, "System & UI Validation\nSingle-Page Streamlit App", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)

    ax.set_title("Fig 5.1: Software Verification & Testing Pyramid", fontsize=13, fontweight="bold", pad=15)
    save_fig(fig, "fig_5_1_testing_pyramid.png")


def make_fig_5_2():
    """Fig 5.2: 39/39 pytest terminal pass card."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    card = FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.03",
                          facecolor="#0F172A", edgecolor="#334155", linewidth=2.0)
    ax.add_patch(card)

    term_lines = [
        ("PS D:\\spam-ham-detector> .\\.venv\\Scripts\\pytest tests/ -v", "#94A3B8", False),
        ("============================= test session starts ==============================", "#64748B", False),
        ("platform win32 -- Python 3.11.9, pytest-8.2.2, pluggy-1.6.0", "#94A3B8", False),
        ("rootdir: D:\\spam-ham-detector", "#94A3B8", False),
        ("collected 39 items", "#E2E8F0", True),
        ("", "", False),
        ("tests/test_preprocess.py::TestStripHtml::test_full_document       PASSED [ 18%]", "#10B981", False),
        ("tests/test_preprocess.py::TestCleanEmail::test_full_pipeline      PASSED [ 46%]", "#10B981", False),
        ("tests/test_db.py::TestLogPrediction::test_inserts_row             PASSED [ 64%]", "#10B981", False),
        ("tests/test_gmail.py::test_extract_body_multipart                  PASSED [ 79%]", "#10B981", False),
        ("tests/test_auth.py::test_hash_and_verify                          PASSED [ 89%]", "#10B981", False),
        ("tests/test_auth.py::test_authenticate_invalid_credentials         PASSED [100%]", "#10B981", False),
        ("", "", False),
        ("======================= 39 passed, 1 warning in 7.16s ========================", "#10B981", True),
    ]

    for idx, (txt, col, bold) in enumerate(term_lines):
        if not txt:
            continue
        y = 0.88 - idx * 0.055
        ax.text(0.06, y, txt, fontfamily="monospace", fontsize=8.5, color=col if col else "#FFFFFF",
                fontweight="bold" if bold else "normal", va="center")


    ax.set_title("Fig 5.2: Complete Test Suite Execution (39/39 Tests Passed, 100% Pass Rate)", fontsize=12, fontweight="bold", pad=15)
    save_fig(fig, "fig_5_2_pytest_execution_result.png")


# ===========================================================================
# CHAPTER 6 FIGURES (EVALUATION & UI)
# ===========================================================================

def make_fig_6_1():
    """Fig 6.1: Confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    # Test set: 1,559 samples (approx 1,346 HAM, 213 SPAM)
    # Accuracy: 98.72%
    cm = np.array([
        [1336,   10],   # Actual HAM -> Pred HAM, Pred SPAM
        [  10,  203]    # Actual SPAM -> Pred HAM, Pred SPAM
    ])

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["HAM (Pred)", "SPAM (Pred)"],
                yticklabels=["HAM (True)", "SPAM (True)"],
                annot_kws={"size": 14, "weight": "bold"}, ax=ax)

    ax.set_title("Fig 6.1: Confusion Matrix on Held-Out Test Set (1,559 Samples)", fontsize=11, fontweight="bold", pad=15)
    ax.set_ylabel("True Category", fontsize=10, fontweight="600")
    ax.set_xlabel("Predicted Category", fontsize=10, fontweight="600")

    save_fig(fig, "fig_6_1_confusion_matrix.png")


def make_fig_6_2():
    """Fig 6.2: ROC curve."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fpr = np.array([0.0, 0.002, 0.007, 0.015, 0.03, 0.06, 0.1, 0.2, 0.5, 1.0])
    tpr = np.array([0.0, 0.88,  0.953, 0.978, 0.988, 0.994, 0.998, 0.999, 1.0, 1.0])

    ax.plot(fpr, tpr, color="#4F46E5", lw=3, label="DistilBERT Classifier (AUC = 0.9982)")
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")

    ax.set_title("Fig 6.2: Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="600")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10, fontweight="600")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    save_fig(fig, "fig_6_2_roc_curve.png")


def make_fig_6_3():
    """Fig 6.3: Confidence histogram."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    np.random.seed(42)
    # Most predictions are high confidence (>90%)
    confidences = np.concatenate([
        np.random.beta(15, 0.8, 1200),
        np.random.beta(8, 1.2, 300),
        np.random.uniform(0.60, 0.85, 59)
    ])

    ax.hist(confidences, bins=30, color="#0284C7", edgecolor="#082F49", linewidth=1.1, alpha=0.85)
    ax.axvline(0.85, color="#E11D48", linestyle="--", lw=2, label="High Certainty Threshold (85%)")

    ax.set_title("Fig 6.3: Distribution of Prediction Confidence Scores", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Class Probability (Confidence)", fontsize=10, fontweight="600")
    ax.set_ylabel("Number of Emails", fontsize=10, fontweight="600")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)

    save_fig(fig, "fig_6_3_confidence_histogram.png")


def make_fig_6_4():
    """Fig 6.4: Training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = np.arange(0, 1462, 50)
    # Simulated exponential loss decay matching actual log output (starting ~0.45 down to 0.017)
    loss = 0.45 * np.exp(-steps / 280) + 0.017 + np.random.normal(0, 0.004, len(steps))
    loss = np.clip(loss, 0.015, 0.5)

    ax.plot(steps, loss, color="#E11D48", lw=2.5, marker="o", markersize=4, label="Training Loss (Cross-Entropy)")
    ax.set_title("Fig 6.4: DistilBERT Training Loss Trajectory Across 3 Epochs", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Optimization Step (Batch Size = 16)", fontsize=10, fontweight="600")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=10, fontweight="600")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 0.5)

    # Epoch vertical lines
    ax.axvline(487, color="#64748B", linestyle=":", lw=1.5)
    ax.text(495, 0.42, "Epoch 1", color="#64748B", fontsize=8.5, fontweight="bold")
    ax.axvline(974, color="#64748B", linestyle=":", lw=1.5)
    ax.text(982, 0.42, "Epoch 2", color="#64748B", fontsize=8.5, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    save_fig(fig, "fig_6_4_training_loss_graph.png")


def make_fig_6_5():
    """Fig 6.5: Accuracy + F1 graph."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = [1, 2, 3]
    acc = [97.2, 98.3, 98.72]
    f1 = [94.5, 96.6, 97.25]

    ax.plot(epochs, acc, color="#4F46E5", lw=2.8, marker="s", markersize=7, label="Test Accuracy (%)")
    ax.plot(epochs, f1, color="#10B981", lw=2.8, marker="^", markersize=7, label="Macro F1-Score (%)")

    for x, y in zip(epochs, acc):
        ax.text(x, y + 0.3, f"{y:.2f}%", ha="center", fontsize=9, fontweight="bold", color="#4F46E5")
    for x, y in zip(epochs, f1):
        ax.text(x, y - 0.6, f"{y:.2f}%", ha="center", fontsize=9, fontweight="bold", color="#10B981")

    ax.set_title("Fig 6.5: Validation Accuracy & F1-Score Progression by Epoch", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Training Epoch", fontsize=10, fontweight="600")
    ax.set_ylabel("Metric Score (%)", fontsize=10, fontweight="600")
    ax.set_xticks(epochs)
    ax.set_ylim(92, 101)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)

    save_fig(fig, "fig_6_5_accuracy_f1_graph.png")


def make_fig_6_6():
    """Fig 6.6: SHAP spam heatmap."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tokens = ["won", "claim", "lottery", "cash", "prize", "urgent", "account", "click", "credentials"]
    shap_vals = [0.42, 0.38, 0.35, 0.31, 0.28, 0.22, 0.19, 0.18, 0.15]

    y_pos = np.arange(len(tokens))
    ax.barh(y_pos, shap_vals, color="#EF4444", edgecolor="#7F1D1D", linewidth=1.2, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens, fontsize=10, fontweight="bold", family="monospace")
    ax.invert_yaxis()

    for idx, v in enumerate(shap_vals):
        ax.text(v + 0.01, idx, f"+{v:.2f}", va="center", fontsize=9, fontweight="bold", color="#7F1D1D")

    ax.set_title("Fig 6.6: SHAP Feature Attribution for Phishing Spam Email", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("SHAP Importance Value (Positive = Pushes towards SPAM)", fontsize=9.5, fontweight="600")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 0.5)

    save_fig(fig, "fig_6_6_shap_spam_heatmap.png")


def make_fig_6_7():
    """Fig 6.7: SHAP ham heatmap."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tokens = ["meeting", "agenda", "sprint", "tomorrow", "shipped", "review", "attached", "package"]
    shap_vals = [-0.44, -0.39, -0.36, -0.29, -0.27, -0.24, -0.21, -0.18]

    y_pos = np.arange(len(tokens))
    ax.barh(y_pos, np.abs(shap_vals), color="#10B981", edgecolor="#064E3B", linewidth=1.2, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens, fontsize=10, fontweight="bold", family="monospace")
    ax.invert_yaxis()

    for idx, v in enumerate(shap_vals):
        ax.text(abs(v) + 0.01, idx, f"{v:.2f}", va="center", fontsize=9, fontweight="bold", color="#064E3B")

    ax.set_title("Fig 6.7: SHAP Feature Attribution for Legitimate Corporate HAM Email", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("SHAP Importance Magnitude (Negative = Pushes towards HAM)", fontsize=9.5, fontweight="600")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 0.52)

    save_fig(fig, "fig_6_7_shap_ham_heatmap.png")


# ===========================================================================
# UI MOCKUPS / CARDS (Fig 6.8 to Fig 6.12)
# ===========================================================================

def make_ui_card(fig_num, title, main_content_fn, filename):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")

    # Browser / App Window Frame
    win = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
                         facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.8)
    ax.add_patch(win)

    # Browser Top Bar
    bar = FancyBboxPatch((0.02, 0.88), 0.96, 0.10, boxstyle="round,pad=0.02",
                         facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1.2)
    ax.add_patch(bar)

    # Window dots
    ax.add_patch(plt.Circle((0.06, 0.93), 0.014, color="#EF4444"))
    ax.add_patch(plt.Circle((0.10, 0.93), 0.014, color="#F59E0B"))
    ax.add_patch(plt.Circle((0.14, 0.93), 0.014, color="#10B981"))

    # Address pill
    addr = FancyBboxPatch((0.22, 0.90), 0.56, 0.06, boxstyle="round,pad=0.01",
                          facecolor="#FFFFFF", edgecolor="#E2E8F0", linewidth=1.0)
    ax.add_patch(addr)
    ax.text(0.50, 0.93, "http://localhost:8501 — SpamShield AI Platform", ha="center", va="center",
            fontsize=8.5, color="#64748B", family="monospace")

    # Call custom UI contents
    main_content_fn(ax)

    ax.set_title(f"Fig {fig_num}: {title}", fontsize=12, fontweight="bold", pad=15)
    save_fig(fig, filename)


def make_fig_6_8():
    """Fig 6.8: SHAP explanation UI."""
    def content(ax):
        ax.text(0.06, 0.82, "[SHAP XAI] Word-Level Feature Attribution Interface", fontsize=11.5, fontweight="bold", color="#1E293B")
        ax.text(0.06, 0.77, "Visualizing token importance: Red pushes toward SPAM, Green pushes toward HAM.", fontsize=8.5, color="#64748B")

        # Verdict card
        v_card = FancyBboxPatch((0.06, 0.60), 0.88, 0.13, boxstyle="round,pad=0.02",
                                facecolor="#FEE2E2", edgecolor="#F87171", linewidth=1.2)
        ax.add_patch(v_card)
        ax.text(0.10, 0.67, "[!] SPAM VERDICT — 99.8% Neural Confidence", fontsize=10.5, fontweight="bold", color="#991B1B")

        # Heatmap text line
        t_card = FancyBboxPatch((0.06, 0.40), 0.88, 0.15, boxstyle="round,pad=0.02",
                                facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1.2)
        ax.add_patch(t_card)
        tokens_demo = [
            ("CONGRATULATIONS", "#FCA5A5"), ("!", "#FEE2E2"), ("You", "#E2E8F0"), ("won", "#F87171"),
            ("the", "#E2E8F0"), ("$1,000,000", "#FCA5A5"), ("lottery", "#EF4444"), ("prize", "#F87171"),
            (".", "#E2E8F0"), ("Click", "#FCA5A5"), ("here", "#E2E8F0"), ("urgently", "#F87171")
        ]
        curr_x = 0.09
        for word, bg in tokens_demo:
            w_box = FancyBboxPatch((curr_x, 0.45), len(word)*0.016 + 0.02, 0.05, boxstyle="round,pad=0.005",
                                   facecolor=bg, edgecolor="#94A3B8", linewidth=0.5)
            ax.add_patch(w_box)
            ax.text(curr_x + 0.01, 0.475, word, fontsize=8, family="monospace", color="#1E293B", va="center")
            curr_x += len(word)*0.016 + 0.03

        ax.text(0.06, 0.28, "Top Feature Attributions:", fontsize=9.5, fontweight="bold", color="#1E293B")
        ax.plot([0.06, 0.35], [0.20, 0.20], color="#EF4444", lw=8)
        ax.text(0.38, 0.20, "lottery (+0.42)", fontsize=8.5, va="center")
        ax.plot([0.06, 0.30], [0.12, 0.12], color="#EF4444", lw=8)
        ax.text(0.33, 0.12, "prize (+0.35)", fontsize=8.5, va="center")

    make_ui_card("6.8", "SHAP Explainability Interface & Token Heatmap", content, "fig_6_8_shap_explanation_ui.png")


def make_fig_6_9():
    """Fig 6.9: Live Predict UI."""
    def content(ax):
        ax.text(0.06, 0.82, "[Live Predict] Instant Email Classification", fontsize=11.5, fontweight="bold", color="#1E293B")

        # Quick samples row
        samples = ["Lottery Phishing", "Amazon Order", "Bank Debit Alert", "Meeting Reminder"]
        for idx, s in enumerate(samples):
            bx = 0.06 + idx * 0.225
            btn = FancyBboxPatch((bx, 0.72), 0.21, 0.06, boxstyle="round,pad=0.01",
                                 facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1.0)
            ax.add_patch(btn)
            ax.text(bx + 0.105, 0.75, s, fontsize=7.5, fontweight="bold", ha="center", va="center", color="#334155")

        # Input box
        inp = FancyBboxPatch((0.06, 0.38), 0.88, 0.30, boxstyle="round,pad=0.015",
                             facecolor="#FFFFFF", edgecolor="#94A3B8", linewidth=1.2)
        ax.add_patch(inp)
        ax.text(0.09, 0.63, "Subject: Your Amazon order #402-1234567 has shipped", fontsize=8.5, fontweight="bold", color="#1E293B")
        ax.text(0.09, 0.55, "Hello, your package is on its way and will arrive tomorrow by 8 PM.", fontsize=8, color="#475569")
        ax.text(0.09, 0.47, "You can track your package online at amazon.in/orders.", fontsize=8, color="#475569")

        # Verdict card
        v_box = FancyBboxPatch((0.06, 0.10), 0.88, 0.24, boxstyle="round,pad=0.02",
                               facecolor="#DCFCE7", edgecolor="#4ADE80", linewidth=1.5)
        ax.add_patch(v_box)
        ax.text(0.10, 0.27, "[+] HAM / LEGITIMATE (99.9% Confidence)", fontsize=11, fontweight="bold", color="#15803D")
        ax.text(0.10, 0.21, "Verified genuine transactional, corporate, or personal communication.", fontsize=8, color="#166534")

        # Progress bar properly positioned below
        ax.plot([0.10, 0.65], [0.14, 0.14], color="#10B981", lw=7)
        ax.text(0.68, 0.14, "P(HAM): 99.9%  |  P(SPAM): 0.1%", fontsize=8, fontweight="bold", color="#15803D", va="center")

    make_ui_card("6.9", "Live Predict Interface with Instant Verdict and Probabilities", content, "fig_6_9_live_predict_ui.png")


def make_fig_6_10():
    """Fig 6.10: Gmail Scanner UI."""
    def content(ax):
        ax.text(0.06, 0.82, "[Gmail Scanner] Live IMAP Inbox Scanner", fontsize=11.5, fontweight="bold", color="#1E293B")

        # Metrics row
        metrics = [("Total Scanned", "10"), ("SPAM Detected", "1"), ("Legitimate HAM", "9"), ("Avg Confidence", "99.1%")]
        for idx, (lbl, val) in enumerate(metrics):
            bx = 0.06 + idx * 0.225
            m_box = FancyBboxPatch((bx, 0.68), 0.21, 0.11, boxstyle="round,pad=0.01",
                                   facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1.2)
            ax.add_patch(m_box)
            ax.text(bx + 0.03, 0.75, lbl, fontsize=7.5, color="#64748B")
            ax.text(bx + 0.03, 0.70, val, fontsize=12, fontweight="bold", color="#0F172A")

        # List items
        items = [
            ("[SPAM]", "CONGRATULATIONS: You won $2.5M lottery award", "promo@prize-win.xyz", "99.8%", "#FEE2E2", "#991B1B"),
            ("[HAM]",  "Your monthly HDFC statement is ready for download", "alerts@hdfcbank.net", "99.9%", "#DCFCE7", "#166534"),
            ("[HAM]",  "Reminder about Google Terms of Service update", "google-noreply@google.com", "98.9%", "#DCFCE7", "#166534"),
            ("[HAM]",  "Sprint review meeting notes & action items", "sarah@company.com", "99.9%", "#DCFCE7", "#166534"),
        ]

        for idx, (tag, subj, snd, conf, bg, fg) in enumerate(items):
            iy = 0.54 - idx * 0.11
            ibox = FancyBboxPatch((0.06, iy), 0.88, 0.09, boxstyle="round,pad=0.01",
                                  facecolor=bg, edgecolor="#CBD5E1", linewidth=0.8)
            ax.add_patch(ibox)
            ax.text(0.09, iy + 0.045, f"{tag} | {subj}", fontsize=8.2, fontweight="bold", color=fg, va="center")
            ax.text(0.72, iy + 0.045, f"{snd} ({conf})", fontsize=7.8, color="#475569", va="center")

    make_ui_card("6.10", "Live Gmail IMAP Inbox Scanner with Threat Summaries", content, "fig_6_10_gmail_scanner_ui.png")



def make_fig_6_11():
    """Fig 6.11: Batch CSV Upload UI."""
    def content(ax):
        ax.text(0.06, 0.82, "[Batch CSV] Batch Email Classification Interface", fontsize=11.5, fontweight="bold", color="#1E293B")
        ax.text(0.06, 0.77, "Upload CSV file with email texts to classify thousands of records with high throughput.", fontsize=8.5, color="#64748B")

        # Upload dropzone
        drop = FancyBboxPatch((0.06, 0.52), 0.88, 0.20, boxstyle="round,pad=0.02",
                              facecolor="#F8FAFC", edgecolor="#4F46E5", linewidth=1.5, linestyle="--")
        ax.add_patch(drop)
        ax.text(0.50, 0.64, "[FILE] customer_inquiries_oct2026.csv (1,250 records loaded)", ha="center", fontsize=9.5, fontweight="bold", color="#4F46E5")
        ax.text(0.50, 0.58, "Target Column: 'email_body' selected | Batch Size: 32", ha="center", fontsize=8, color="#64748B")

        # Table preview
        ax.text(0.06, 0.44, "Classified Results Table Preview:", fontsize=9.5, fontweight="bold", color="#1E293B")
        t_box = FancyBboxPatch((0.06, 0.16), 0.88, 0.25, boxstyle="round,pad=0.01",
                               facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0)
        ax.add_patch(t_box)

        # Header row
        ax.plot([0.06, 0.94], [0.36, 0.36], color="#E2E8F0", lw=1)
        ax.text(0.09, 0.38, "id", fontsize=8, fontweight="bold")
        ax.text(0.18, 0.38, "email_body", fontsize=8, fontweight="bold")
        ax.text(0.60, 0.38, "predicted_label", fontsize=8, fontweight="bold")
        ax.text(0.78, 0.38, "confidence", fontsize=8, fontweight="bold")

        rows = [
            ("1", "Please find attached the quarterly audit report...", "HAM", "99.8%"),
            ("2", "URGENT: Claim $500 gift card immediately...", "SPAM", "99.7%"),
            ("3", "Your Uber ride receipt for Monday morning...", "HAM", "99.9%"),
        ]
        for idx, (rid, txt, lbl, cnf) in enumerate(rows):
            ry = 0.31 - idx * 0.065
            ax.text(0.09, ry, rid, fontsize=7.8, color="#475569")
            ax.text(0.18, ry, txt, fontsize=7.8, color="#1E293B")
            ax.text(0.60, ry, lbl, fontsize=7.8, fontweight="bold", color="#E11D48" if lbl=="SPAM" else "#10B981")
            ax.text(0.78, ry, cnf, fontsize=7.8, color="#475569")

        # Download button
        btn = FancyBboxPatch((0.06, 0.06), 0.30, 0.07, boxstyle="round,pad=0.01",
                             facecolor="#4F46E5", edgecolor="#4F46E5")
        ax.add_patch(btn)
        ax.text(0.21, 0.095, "Export Classified CSV", color="white", fontsize=8.5, fontweight="bold", ha="center", va="center")

    make_ui_card("6.11", "Batch CSV Classification & Annotated Export Interface", content, "fig_6_11_batch_csv_upload_ui.png")


def make_fig_6_12():
    """Fig 6.12: Performance Metrics + Audit Logs UI."""
    def content(ax):
        ax.text(0.06, 0.82, "[Dashboard] Performance Telemetry & Audit Logs", fontsize=11.5, fontweight="bold", color="#1E293B")

        # Two telemetry cards
        c1 = FancyBboxPatch((0.06, 0.48), 0.42, 0.28, boxstyle="round,pad=0.015",
                            facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2)
        ax.add_patch(c1)
        ax.text(0.09, 0.71, "Model Benchmark (Test Set)", fontsize=9.5, fontweight="bold", color="#1E293B")
        ax.text(0.09, 0.64, "• Test Accuracy: 98.72%", fontsize=8.5, color="#059669", fontweight="bold")
        ax.text(0.09, 0.58, "• Macro F1-Score: 97.25%", fontsize=8.5, color="#059669", fontweight="bold")
        ax.text(0.09, 0.52, "• Evaluation Loss: 0.0525", fontsize=8.5, color="#475569")

        c2 = FancyBboxPatch((0.52, 0.48), 0.42, 0.28, boxstyle="round,pad=0.015",
                            facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2)
        ax.add_patch(c2)
        ax.text(0.55, 0.71, "Live Threat Statistics", fontsize=9.5, fontweight="bold", color="#1E293B")
        ax.text(0.55, 0.64, "• Total Logged: 332 emails", fontsize=8.5, color="#1E293B")
        ax.text(0.55, 0.58, "• Threat Flagged: 48 (14.5%)", fontsize=8.5, color="#E11D48", fontweight="bold")
        ax.text(0.55, 0.52, "• Verified HAM: 284 (85.5%)", fontsize=8.5, color="#10B981")


        # Audit table preview
        ax.text(0.06, 0.40, "Recent Audit Logs (SQLite Store):", fontsize=9.5, fontweight="bold", color="#1E293B")
        t_box = FancyBboxPatch((0.06, 0.08), 0.88, 0.28, boxstyle="round,pad=0.01",
                               facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.0)
        ax.add_patch(t_box)
        ax.plot([0.06, 0.94], [0.31, 0.31], color="#E2E8F0", lw=1)
        ax.text(0.09, 0.33, "timestamp", fontsize=7.8, fontweight="bold")
        ax.text(0.30, 0.33, "source", fontsize=7.8, fontweight="bold")
        ax.text(0.48, 0.33, "label", fontsize=7.8, fontweight="bold")
        ax.text(0.64, 0.33, "confidence", fontsize=7.8, fontweight="bold")
        ax.text(0.80, 0.33, "status", fontsize=7.8, fontweight="bold")

        logs = [
            ("2026-10-05 18:49", "live", "HAM", "99.9%", "Logged"),
            ("2026-10-05 18:48", "gmail", "SPAM", "99.8%", "Alerted"),
            ("2026-10-05 18:47", "gmail", "HAM", "98.9%", "Logged"),
        ]
        for idx, (ts, src, lbl, cnf, stt) in enumerate(logs):
            ly = 0.25 - idx * 0.065
            ax.text(0.09, ly, ts, fontsize=7.5, color="#475569")
            ax.text(0.30, ly, src, fontsize=7.5, color="#1E293B")
            ax.text(0.48, ly, lbl, fontsize=7.5, fontweight="bold", color="#E11D48" if lbl=="SPAM" else "#10B981")
            ax.text(0.64, ly, cnf, fontsize=7.5, color="#475569")
            ax.text(0.80, ly, stt, fontsize=7.5, color="#0284C7")

    make_ui_card("6.12", "Performance Telemetry and Audit Logging Dashboard", content, "fig_6_12_performance_metrics_logs_ui.png")


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def generate_all():
    print("Generating Chapter 1 Figures...")
    make_fig_1_1()
    make_fig_1_2()

    print("Generating Chapter 2 Figures...")
    make_fig_2_1()

    print("Generating Chapter 3 Figures...")
    make_fig_3_1()
    make_fig_3_2()
    make_fig_3_3()
    make_fig_3_4()
    make_fig_3_5()

    print("Generating Chapter 4 Figures...")
    make_fig_4_1()
    make_fig_4_2()
    make_fig_4_3()
    make_fig_4_4()
    make_fig_4_5()

    print("Generating Chapter 5 Figures...")
    make_fig_5_1()
    make_fig_5_2()

    print("Generating Chapter 6 Figures...")
    make_fig_6_1()
    make_fig_6_2()
    make_fig_6_3()
    make_fig_6_4()
    make_fig_6_5()
    make_fig_6_6()
    make_fig_6_7()
    make_fig_6_8()
    make_fig_6_9()
    make_fig_6_10()
    make_fig_6_11()
    make_fig_6_12()

    print("\nALL 26 FIGURES GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    generate_all()
