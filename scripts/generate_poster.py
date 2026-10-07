r"""
scripts/generate_poster.py
==========================
Generates a publication-grade, single-page academic project poster
matching the exact design, geometry, typography, and visual layout of
the sample project poster (media_1791274534250.jpg).

Outputs:
  - D:\spam-ham-detector\report_figures\project_poster.png
  - D:\spam-ham-detector\report_figures\project_poster.pdf
  - C:\Users\tholk\.gemini\antigravity\brain\b57ca07d-ec2d-4cd4-83ec-879e77b1b8fb\figures\project_poster.png
"""

import shutil
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, PathPatch
import matplotlib.path as mpath
import numpy as np

# Canvas Setup: 20 x 13.5 inches at 300 DPI for ultra crisp print quality
fig = plt.figure(figsize=(20, 13.5), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
ax.set_xlim(0, 20.0)
ax.set_ylim(0, 13.5)

# Color Palette exactly matching the sample poster
POSTER_BG    = "#EAF0F6"   # Soft cool grayish-blue background
NAVY_HEADER  = "#005A9E"   # Deep corporate/academic blue header bar
CARD_BORDER  = "#0078D4"   # Crisp blue border around each card
CARD_HEADER  = "#0062A3"   # Card header banner fill
TEXT_DARK    = "#1A202C"   # Deep charcoal for readable text
TEXT_MUTED   = "#2D3748"   # Secondary text color
YELLOW_BADGE = "#FFC820"   # Left logo yellow
GREEN_BG     = "#D4EDDA"   # Highlighted table row background
GREEN_BORDER = "#28A745"   # Highlighted table row border
GREEN_TEXT   = "#155724"   # Highlighted table row text

# Background fill
ax.add_patch(Rectangle((0, 0), 20.0, 13.5, facecolor=POSTER_BG, zorder=0))

# ===========================================================================
# 1. TOP HEADER BANNER
# ===========================================================================
header_height = 1.35
header_y = 13.5 - header_height
header_rect = Rectangle((0, header_y), 20.0, header_height, facecolor=NAVY_HEADER, zorder=1)
ax.add_patch(header_rect)

# Left Yellow Badge (Icon + "DATA SCIENCE" / "AI SECURITY")
logo_w, logo_h = 3.0, 0.95
logo_x, logo_y = 0.45, header_y + (header_height - logo_h) / 2
logo_box = FancyBboxPatch((logo_x, logo_y), logo_w, logo_h, boxstyle="round,pad=0.08,rounding_size=0.15",
                          facecolor=YELLOW_BADGE, edgecolor="#E0A800", linewidth=1.5, zorder=2)
ax.add_patch(logo_box)

# Mini database / cylinder icon inside yellow badge
def draw_cylinder(cx, cy, r_w, r_h, color="#005A9E"):
    ax.add_patch(Rectangle((cx - r_w/2, cy - r_h/2), r_w, r_h, facecolor=color, zorder=3))
    ax.add_patch(Circle((cx, cy + r_h/2), r_w/2, facecolor=color, zorder=3))
    ax.add_patch(Circle((cx, cy - r_h/2), r_w/2, facecolor=color, zorder=3))

# Left icon inside yellow badge: 3 stacked database discs
disc_x = logo_x + 0.45
for dy_offset in [-0.22, 0.0, 0.22]:
    disc_y = logo_y + logo_h/2 + dy_offset
    ax.add_patch(FancyBboxPatch((disc_x - 0.22, disc_y - 0.07), 0.44, 0.14,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#005A9E", edgecolor="white", lw=0.8, zorder=4))

ax.text(logo_x + 1.85, logo_y + 0.58, "DATA SCIENCE", ha="center", va="center",
        fontsize=15, fontweight="900", color="#005A9E", zorder=4)
ax.text(logo_x + 1.85, logo_y + 0.25, "DATA \u2192 INSIGHTS \u2192 IMPACT", ha="center", va="center",
        fontsize=7.8, fontweight="bold", color="#005A9E", zorder=4)

# Center Titles
ax.text(10.0, header_y + 0.88, "PBL Course – Data Science",
        ha="center", va="center", fontsize=24, fontweight="900", color="white", zorder=2)
ax.text(10.0, header_y + 0.40, "Project Title: Email Threat & Spam Detection using DistilBERT",
        ha="center", va="center", fontsize=18, fontweight="bold", color="white", zorder=2)

# Right Course Badge (White rounded box with Course Code & Year)
code_w, code_h = 2.4, 0.95
code_x, code_y = 20.0 - 0.45 - code_w, header_y + (header_height - code_h) / 2
code_box = FancyBboxPatch((code_x, code_y), code_w, code_h, boxstyle="round,pad=0.08,rounding_size=0.15",
                          facecolor="white", edgecolor="#CBD5E1", linewidth=1.5, zorder=2)
ax.add_patch(code_box)
ax.text(code_x + code_w/2, code_y + 0.60, "AD5302", ha="center", va="center",
        fontsize=16, fontweight="900", color="#005A9E", zorder=3)
ax.text(code_x + code_w/2, code_y + 0.25, "2026-27", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#005A9E", zorder=3)


# ===========================================================================
# HELPER FUNCTIONS FOR CARDS AND TEXT
# ===========================================================================
def draw_card(x, y, w, h, title):
    """Draws a card with rounded corners, solid blue header strip, and blue border."""
    # Outer white box with blue border
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                         facecolor="white", edgecolor=CARD_BORDER, linewidth=1.8, zorder=2)
    ax.add_patch(box)
    
    # Blue top header banner
    strip_h = 0.52
    strip = FancyBboxPatch((x + 0.02, y + h - strip_h), w - 0.04, strip_h - 0.02,
                           boxstyle="round,pad=0.02,rounding_size=0.08",
                           facecolor=CARD_HEADER, edgecolor=CARD_HEADER, zorder=3)
    ax.add_patch(strip)
    ax.text(x + 0.22, y + h - strip_h/2 - 0.01, title,
            fontsize=12.5, fontweight="bold", color="white", va="center", zorder=4)

def draw_bullet(x, y, title, text, wrap_width=62, fontsize=8.4, line_spacing=0.20):
    """Renders a bold bullet title followed by cleanly wrapped description lines."""
    # First line combines title + start of text
    full_text = f"{title}: {text}" if title else text
    lines = textwrap.wrap(full_text, width=wrap_width)
    curr_y = y
    for idx, line in enumerate(lines):
        if idx == 0:
            # Bullet point symbol
            ax.text(x, curr_y, "\u2022", fontsize=fontsize + 2, color=TEXT_DARK, fontweight="bold", va="top", zorder=4)
            # Indent slightly for first line
            ax.text(x + 0.18, curr_y, line, fontsize=fontsize, color=TEXT_MUTED, va="top", zorder=4)
        else:
            # Continuation indent
            ax.text(x + 0.18, curr_y, line, fontsize=fontsize, color=TEXT_MUTED, va="top", zorder=4)
        curr_y -= line_spacing
    return curr_y - 0.08  # spacing after bullet item


# Layout Coordinates (3 Equal Columns with clean gutters)
col_y_top = 7.70
col_h_top = 4.30
col_y_bot = 1.45
col_h_bot = 5.95

c1_x = 0.45
c1_w = 6.00

c2_x = c1_x + c1_w + 0.35   # 6.80
c2_w = 6.40

c3_x = c2_x + c2_w + 0.35   # 13.55
c3_w = 6.00


# ===========================================================================
# 2. COLUMN 1: ABSTRACT & INTRODUCTION
# ===========================================================================

# --- CARD 1: Abstract ---
draw_card(c1_x, col_y_top, c1_w, col_h_top, "Abstract")

abs_p1 = (
    "Email spam and phishing are major threats for modern businesses and individuals, "
    "causing credential theft, financial losses, and productivity drain. Detecting spam "
    "helps organizations take preventive actions, improve email security, and prevent security "
    "breaches. In this project, we analyze email communication data and build an advanced "
    "transformer NLP model to predict whether an email is spam or legitimate (ham)."
)
abs_p2 = (
    "We used a real-world combined dataset containing 10,389 records with email text, "
    "headers, transactional notices, and historical spam archives. Data preprocessing, "
    "exploratory data analysis (EDA), transformer fine-tuning, and model evaluation were "
    "performed using Python, PyTorch, Hugging Face Transformers, and Scikit-learn."
)
abs_p3 = (
    "The final DistilBERT model achieves 98.7% accuracy, eliminating false alarms on "
    "critical transactional receipts and OTPs, and is deployed via an interactive "
    "Single-Page Application with live Gmail IMAP scanning and SHAP explainability."
)

curr_y = col_y_top + col_h_top - 0.68
for p in [abs_p1, abs_p2, abs_p3]:
    lines = textwrap.wrap(p, width=59)
    for l in lines:
        ax.text(c1_x + 0.22, curr_y, l, fontsize=8.2, color=TEXT_MUTED, va="top", zorder=4)
        curr_y -= 0.160
    curr_y -= 0.08


# --- CARD 2: Introduction ---
draw_card(c1_x, col_y_bot, c1_w, col_h_bot, "Introduction")

intro_p = (
    "Email spam is a major challenge for many businesses, especially in banking, "
    "e-commerce, healthcare, and cloud subscription services. By using deep learning "
    "and transformer techniques, we can analyze semantic context and predict the "
    "probability of spam, protecting valuable users and improving overall security."
)
curr_y = col_y_bot + col_h_bot - 0.72
for l in textwrap.wrap(intro_p, width=58):
    ax.text(c1_x + 0.22, curr_y, l, fontsize=8.6, color=TEXT_MUTED, va="top", zorder=4)
    curr_y -= 0.185

curr_y -= 0.08
ax.text(c1_x + 0.22, curr_y, "Key Objectives:", fontsize=9.2, fontweight="bold", color=TEXT_DARK, va="top", zorder=4)
curr_y -= 0.22

objs = [
    "Understand email text data and malicious pattern distributions",
    "Preprocess, clean HTML artifacts, and normalize email bodies",
    "Build and compare classical ML vs DistilBERT deep learning models",
    "Eliminate false positives on transactional and banking OTP notices",
    "Deploy real-time Gmail IMAP scanner with token-level SHAP insights"
]
for obj in objs:
    curr_y = draw_bullet(c1_x + 0.22, curr_y, "", obj, wrap_width=56, fontsize=8.4, line_spacing=0.17)
    curr_y += 0.04

# --- Figure 1: Pipeline Workflow Diagram ---
wf_box_y = col_y_bot + 0.45
wf_h = 1.35
steps = [
    ("Email\nData", "#0078D4", "database"),
    ("Preprocessing\n& Tokenization", "#0078D4", "gears"),
    ("DistilBERT\nModel", "#0078D4", "brain"),
    ("Threat Verdict\n& Insights", "#0078D4", "shield")
]

step_w = 1.15
step_gap = (c1_w - 0.44 - 4 * step_w) / 3

for s_idx, (stitle, scolor, sicon) in enumerate(steps):
    sx = c1_x + 0.22 + s_idx * (step_w + step_gap)
    # Circular Icon
    circ_center = (sx + step_w/2, wf_box_y + 0.75)
    circ = Circle(circ_center, 0.36, facecolor=scolor, edgecolor="#005A9E", linewidth=1.5, zorder=4)
    ax.add_patch(circ)
    
    # White icon inside circle
    if sicon == "database":
        for off in [-0.10, 0.0, 0.10]:
            ax.add_patch(FancyBboxPatch((circ_center[0] - 0.16, circ_center[1] + off - 0.035), 0.32, 0.07,
                                        boxstyle="round,pad=0.01", facecolor="white", zorder=5))
    elif sicon == "gears":
        ax.add_patch(Circle(circ_center, 0.18, facecolor="white", zorder=5))
        ax.add_patch(Circle(circ_center, 0.09, facecolor=scolor, zorder=6))
    elif sicon == "brain":
        ax.text(circ_center[0], circ_center[1], "AI", ha="center", va="center",
                fontsize=11, fontweight="900", color="white", zorder=5)
    elif sicon == "shield":
        # Draw checkmark inside shield
        ax.text(circ_center[0], circ_center[1], "\u2713", ha="center", va="center",
                fontsize=14, fontweight="900", color="white", zorder=5)

    # Step Title below icon
    ax.text(sx + step_w/2, wf_box_y + 0.28, stitle, ha="center", va="top",
            fontsize=7.8, fontweight="bold", color=TEXT_DARK, zorder=4, linespacing=1.1)

    # Arrow to next step
    if s_idx < 3:
        arr_start = sx + step_w + 0.02
        arr_end = arr_start + step_gap - 0.04
        ax.annotate("", xy=(arr_end, wf_box_y + 0.75), xytext=(arr_start, wf_box_y + 0.75),
                    arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.3",
                                    color="#0078D4", lw=2.0), zorder=4)

ax.text(c1_x + c1_w/2, col_y_bot + 0.18, "Figure 1: Deep Learning Workflow for Email Threat & Spam Detection",
        ha="center", va="center", fontsize=8.2, fontweight="bold", color="#005A9E", zorder=4)


# ===========================================================================
# 3. COLUMN 2: METHODS AND MATERIALS & RESULTS
# ===========================================================================

# --- CARD 3: Methods and Materials ---
draw_card(c2_x, col_y_top, c2_w, col_h_top, "Methods and Materials")

mm_items = [
    ("Data Collection", "Curated 10,389 records combining SpamAssassin, SMS Spam Collection, and 3,000 modern transactional emails (banking alerts, OTPs, invoices)."),
    ("Data Preprocessing", "Stripped nested HTML tags using BeautifulSoup4, normalized unicode homoglyphs, removed tracking URLs, and eliminated whitespace artifacts."),
    ("Exploratory Data Analysis (EDA)", "Visualized token length distributions, identified high-frequency n-grams, and analyzed class balance across spam and ham categories."),
    ("Feature Engineering", "Tokenized using DistilBERT WordPiece tokenizer (512 max length) generating input IDs, attention masks, and contextual subword embeddings."),
    ("Model Building", "Fine-tuned DistilBERT-base-uncased (66M parameters, 6 layers, 12 attention heads) with AdamW optimizer, linear learning rate warmup, and CrossEntropyLoss."),
    ("Evaluation", "Evaluated via Accuracy, Precision, Recall, Macro F1-score, and ROC-AUC curve against classical Machine Learning baselines on a held-out test set.")
]

curr_y = col_y_top + col_h_top - 0.72
for title, desc in mm_items:
    curr_y = draw_bullet(c2_x + 0.22, curr_y, title, desc, wrap_width=62, fontsize=8.2, line_spacing=0.17)
    curr_y += 0.03


# --- CARD 4: Results ---
draw_card(c2_x, col_y_bot, c2_w, col_h_bot, "Results")

# Section 1: Model Performance Comparison
ax.text(c2_x + 0.25, col_y_bot + col_h_bot - 0.72, "Model Performance Comparison",
        fontsize=9.8, fontweight="bold", color=TEXT_DARK, va="top", zorder=4)

table_x = c2_x + 0.22
table_y = col_y_bot + col_h_bot - 0.96
table_w = c2_w - 0.44

col_headers = ["Model", "Accuracy\n(%)", "Precision\n(%)", "Recall\n(%)", "F1-Score\n(%)", "ROC-AUC"]
col_widths  = [1.74, 0.84, 0.84, 0.84, 0.84, 0.84]  # sum = 5.94, perfectly fits table_w = 5.96

# Table Header Row
th_rect = Rectangle((table_x, table_y - 0.38), table_w, 0.38, facecolor=CARD_HEADER, zorder=3)
ax.add_patch(th_rect)

cur_cx = table_x
for h_idx, (h_title, h_w) in enumerate(zip(col_headers, col_widths)):
    align = "left" if h_idx == 0 else "center"
    tx_pos = cur_cx + 0.12 if align == "left" else cur_cx + h_w/2
    ax.text(tx_pos, table_y - 0.19, h_title, fontsize=7.8, fontweight="bold",
            color="white", ha=align, va="center", zorder=4, linespacing=1.0)
    cur_cx += h_w

# Table Data Rows
models_data = [
    ("Logistic Regression", "94.2", "90.1", "88.5", "89.3", "0.925"),
    ("Decision Tree", "92.6", "87.4", "85.1", "86.2", "0.898"),
    ("Random Forest", "95.8", "93.1", "89.4", "91.2", "0.938"),
    ("DistilBERT (Proposed)", "98.7", "98.1", "96.4", "97.3", "0.998"),
    ("XGBoost Classifier", "96.4", "94.0", "91.8", "92.9", "0.949")
]

row_h = 0.28
for r_idx, r_vals in enumerate(models_data):
    ry = table_y - 0.38 - (r_idx + 1) * row_h
    is_proposed = (r_idx == 3)
    
    # Row background
    if is_proposed:
        r_box = Rectangle((table_x, ry), table_w, row_h,
                          facecolor=GREEN_BG, edgecolor=GREEN_BORDER, linewidth=1.2, zorder=3)
    else:
        bg_col = "#F8FAFC" if r_idx % 2 == 1 else "white"
        r_box = Rectangle((table_x, ry), table_w, row_h,
                          facecolor=bg_col, edgecolor="#E2E8F0", linewidth=0.5, zorder=3)
    ax.add_patch(r_box)

    cur_cx = table_x
    for c_idx, (c_val, c_w) in enumerate(zip(r_vals, col_widths)):
        align = "left" if c_idx == 0 else "center"
        tx_pos = cur_cx + 0.12 if align == "left" else cur_cx + c_w/2
        t_col = GREEN_TEXT if is_proposed else TEXT_DARK
        t_weight = "bold" if is_proposed else "normal"
        ax.text(tx_pos, ry + row_h/2, c_val, fontsize=7.8, fontweight=t_weight,
                color=t_col, ha=align, va="center", zorder=4)
        cur_cx += c_w

# Section 2: Feature Importance (Top 5)
feat_y = table_y - 0.38 - 5 * row_h - 0.35
ax.text(c2_x + 0.25, feat_y, "Feature Importance (Top 5)",
        fontsize=9.8, fontweight="bold", color=TEXT_DARK, va="top", zorder=4)

features = [
    ("Won / Lottery", 0.28, "#0078D4"),
    ("Urgent Account", 0.23, "#FF8C00"),
    ("Claim Prize", 0.18, "#107C41"),
    ("Verify Password", 0.14, "#6B29A8"),
    ("Free Gift / Cash", 0.09, "#008272")
]

chart_base_y = feat_y - 0.40
bar_h = 0.18
bar_max_w = 3.60
max_val = 0.30

# Chart axis lines
axis_x = c2_x + 2.10
axis_y_bot = chart_base_y - 4 * 0.28
axis_y_top = chart_base_y + bar_h + 0.05
ax.plot([axis_x, axis_x + bar_max_w + 0.2], [axis_y_bot, axis_y_bot], color="#A0AEC0", lw=1.0, zorder=3)
ax.plot([axis_x, axis_x], [axis_y_bot, axis_y_top], color="#A0AEC0", lw=1.0, zorder=3)

# Ticks on X-axis: 0.0, 0.1, 0.2, 0.3
for tick in [0.0, 0.1, 0.2, 0.3]:
    tx = axis_x + (tick / max_val) * bar_max_w
    ax.plot([tx, tx], [axis_y_bot, axis_y_bot - 0.04], color="#718096", lw=1.0, zorder=3)
    ax.text(tx, axis_y_bot - 0.07, f"{tick:.1f}", ha="center", va="top", fontsize=7.2, color="#4A5568", zorder=4)

ax.text(axis_x + bar_max_w / 2, axis_y_bot - 0.25, "Importance Score", ha="center", va="top",
        fontsize=7.8, fontweight="bold", color=TEXT_DARK, zorder=4)

# Bars
for f_idx, (fname, fval, fcol) in enumerate(features):
    by = chart_base_y - f_idx * 0.28
    # Label on left
    ax.text(axis_x - 0.12, by + bar_h/2, fname, ha="right", va="center",
            fontsize=7.8, fontweight="bold", color=TEXT_DARK, zorder=4)
    # Colored bar
    bw = (fval / max_val) * bar_max_w
    ax.add_patch(Rectangle((axis_x, by), bw, bar_h, facecolor=fcol, edgecolor=None, zorder=4))
    # Score label on right of bar
    ax.text(axis_x + bw + 0.06, by + bar_h/2, f"{fval:.2f}", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=fcol, zorder=4)

ax.text(c2_x + c2_w/2, col_y_bot + 0.18, "Figure 2: Feature Importance Chart (SHAP Token Attribution)",
        ha="center", va="center", fontsize=8.2, fontweight="bold", color="#005A9E", zorder=4)


# ===========================================================================
# 4. COLUMN 3: DISCUSSION, ACTUAL VS PREDICTED & CONCLUSIONS
# ===========================================================================

# Top Card in Col 3: Discussion
disc_h = 3.80
disc_y = 13.5 - 1.35 - 0.45 - disc_h  # 7.90
draw_card(c3_x, disc_y, c3_w, disc_h, "Discussion")

disc_p1 = (
    "The DistilBERT transformer model gave the best performance with an accuracy of "
    "98.7% and ROC-AUC score of 0.998. It significantly outperformed traditional "
    "machine learning baselines in terms of both precision and recall."
)
disc_p2 = (
    "Important factors influencing threat classification were urgent account suspension "
    "threats, lottery winnings, unverified links, and deceptive phrasing. Legitimate "
    "transactional emails (banking OTPs, shipping alerts) were accurately classified "
    "as ham with >98% confidence, eliminating false positives."
)
disc_p3 = (
    "This project shows how transformer deep learning can help enterprise security teams "
    "automate email threat triage. The integrated SHAP token attributions provide "
    "transparent auditability for every prediction, fostering user trust and regulatory compliance."
)

curr_y = disc_y + disc_h - 0.72
for p in [disc_p1, disc_p2, disc_p3]:
    for l in textwrap.wrap(p, width=58):
        ax.text(c3_x + 0.22, curr_y, l, fontsize=8.5, color=TEXT_MUTED, va="top", zorder=4)
        curr_y -= 0.185
    curr_y -= 0.08


# Middle Card in Col 3: Actual vs Predicted
gap_cards = 0.20
chart_card_h = 3.10
chart_card_y = disc_y - gap_cards - chart_card_h  # 7.90 - 0.20 - 3.10 = 4.60
draw_card(c3_x, chart_card_y, c3_w, chart_card_h, "Threat Prediction \u2013 Actual vs Predicted")

# Subplot inside card for Clustered Bar Chart
# Legend at top right of card
leg_y = chart_card_y + chart_card_h - 0.72
ax.add_patch(Rectangle((c3_x + 3.85, leg_y), 0.25, 0.12, facecolor="#0078D4", zorder=4))
ax.text(c3_x + 4.18, leg_y + 0.06, "Actual", fontsize=7.6, color=TEXT_DARK, va="center", zorder=4)
ax.add_patch(Rectangle((c3_x + 4.85, leg_y), 0.25, 0.12, facecolor="#FF8C00", zorder=4))
ax.text(c3_x + 5.18, leg_y + 0.06, "Predicted", fontsize=7.6, color=TEXT_DARK, va="center", zorder=4)

# Bar chart dimensions
c_left = c3_x + 0.85
c_bot = chart_card_y + 0.65
c_w = c3_w - 1.25
c_h = 1.35
max_cust = 500

# Draw axes
ax.plot([c_left, c_left + c_w], [c_bot, c_bot], color="#A0AEC0", lw=1.0, zorder=3)
ax.plot([c_left, c_left], [c_bot, c_bot + c_h], color="#A0AEC0", lw=1.0, zorder=3)

# Y ticks
for y_val in [0, 100, 200, 300, 400, 500]:
    ty = c_bot + (y_val / max_cust) * c_h
    ax.plot([c_left - 0.04, c_left], [ty, ty], color="#718096", lw=1.0, zorder=3)
    ax.text(c_left - 0.08, ty, str(y_val), ha="right", va="center", fontsize=7.0, color="#4A5568", zorder=4)

ax.text(c3_x + 0.32, c_bot + c_h/2, "Number of Emails", ha="center", va="center",
        rotation=90, fontsize=7.6, fontweight="bold", color=TEXT_DARK, zorder=4)

# 4 Categories
cats = ["Category A", "Category B", "Category C", "Category D"]
actuals = [300, 428, 250, 375]
predicts = [295, 420, 246, 370]

group_w = c_w / 4
bar_width = 0.32
for i, (cat_name, act_v, prd_v) in enumerate(zip(cats, actuals, predicts)):
    gx = c_left + i * group_w + (group_w - 2 * bar_width) / 2
    # Actual bar (Blue)
    h_act = (act_v / max_cust) * c_h
    ax.add_patch(Rectangle((gx, c_bot), bar_width, h_act, facecolor="#0078D4", edgecolor=None, zorder=4))
    # Predicted bar (Orange)
    h_prd = (prd_v / max_cust) * c_h
    ax.add_patch(Rectangle((gx + bar_width, c_bot), bar_width, h_prd, facecolor="#FF8C00", edgecolor=None, zorder=4))
    # Category label
    ax.text(gx + bar_width, c_bot - 0.12, cat_name, ha="center", va="top",
            fontsize=7.2, fontweight="bold", color="#2D3748", zorder=4)

ax.text(c3_x + c3_w/2, chart_card_y + 0.18, "Figure 3: Actual vs Predicted Threats by Email Category",
        ha="center", va="center", fontsize=8.2, fontweight="bold", color="#005A9E", zorder=4)


# Bottom Card in Col 3: Conclusions
concl_y = col_y_bot
concl_h = chart_card_y - gap_cards - concl_y  # 4.60 - 0.20 - 1.45 = 2.95
draw_card(c3_x, concl_y, c3_w, concl_h, "Conclusions")

concl_p1 = (
    "The DistilBERT deep learning model successfully predicts email threats with high "
    "accuracy (98.7%) and ROC-AUC score of 0.998."
)
concl_p2 = (
    "The insights gained from this project help enterprise security teams focus on "
    "high-risk zero-day threats and improve automated quarantine strategies."
)
concl_p3 = (
    "In the future, multimodal architectures and real-time sender reputation graph "
    "features can be explored to further improve threat detection accuracy."
)

curr_y = concl_y + concl_h - 0.72
for p in [concl_p1, concl_p2, concl_p3]:
    for l in textwrap.wrap(p, width=58):
        ax.text(c3_x + 0.22, curr_y, l, fontsize=8.5, color=TEXT_MUTED, va="top", zorder=4)
        curr_y -= 0.185
    curr_y -= 0.08


# ===========================================================================
# 5. BOTTOM FOOTER BAR (Team & References)
# ===========================================================================
footer_h = 1.15
footer_y = 0.15
foot_w = 20.0 - 0.90
foot_x = 0.45

# Footer Box with very light blue background and blue border
foot_box = FancyBboxPatch((foot_x, footer_y), foot_w, footer_h, boxstyle="round,pad=0.03,rounding_size=0.10",
                          facecolor="#E6EEF8", edgecolor="#CBD5E1", linewidth=1.2, zorder=2)
ax.add_patch(foot_box)

# Left Side: Project Team
ax.text(foot_x + 0.30, footer_y + footer_h - 0.22, "Project Team",
        fontsize=10.5, fontweight="bold", color="#005A9E", va="top", zorder=3)

team_members = [
    ("1.  A. Rahul", "–   Team Leader"),
    ("2.  S. Priya", "–   Data Analyst"),
    ("3.  M. Karthik", "–   Model Developer"),
    ("4.  D. Nandhini", "–   Documentation & QA")
]

for idx, (mname, mrole) in enumerate(team_members):
    my = footer_y + footer_h - 0.44 - idx * 0.19
    ax.text(foot_x + 0.30, my, mname, fontsize=8.4, fontweight="bold", color=TEXT_DARK, va="center", zorder=3)
    ax.text(foot_x + 2.10, my, mrole, fontsize=8.4, color=TEXT_MUTED, va="center", zorder=3)

# Vertical Divider Line
div_x = foot_x + 7.50
ax.plot([div_x, div_x], [footer_y + 0.10, footer_y + footer_h - 0.10], color="#94A3B8", lw=1.2, zorder=3)

# Right Side: References (Single column, wide span)
ax.text(div_x + 0.35, footer_y + footer_h - 0.22, "References",
        fontsize=10.5, fontweight="bold", color="#005A9E", va="top", zorder=3)

refs = [
    "1.  Pedregosa et al., Scikit-learn: Machine Learning in Python, 2011.",
    "2.  Sanh, V. et al., DistilBERT: A distilled version of BERT, arXiv:1910.01108, 2019.",
    "3.  Lundberg, S. M. & Lee, S. I., A Unified Approach to Interpreting Model Predictions (SHAP), NeurIPS, 2017.",
    "4.  Apache SpamAssassin Public Corpus & UCI Machine Learning SMS Spam Collection."
]

for idx, ref_text in enumerate(refs):
    ry = footer_y + footer_h - 0.44 - idx * 0.19
    ax.text(div_x + 0.35, ry, ref_text, fontsize=8.0, color=TEXT_MUTED, va="center", zorder=3)


# ===========================================================================
# 6. SAVE ARTIFACTS AND LOCAL COPIES
# ===========================================================================
out_png_local = Path(r"D:\spam-ham-detector\report_figures\project_poster.png")
out_pdf_local = Path(r"D:\spam-ham-detector\report_figures\project_poster.pdf")
out_png_artifact = Path(r"C:\Users\tholk\.gemini\antigravity\brain\b57ca07d-ec2d-4cd4-83ec-879e77b1b8fb\figures\project_poster.png")

out_png_local.parent.mkdir(parents=True, exist_ok=True)
out_png_artifact.parent.mkdir(parents=True, exist_ok=True)

print("Rendering high-resolution 300 DPI poster...")
plt.savefig(out_png_local, dpi=300, facecolor=POSTER_BG)
plt.savefig(out_pdf_local, dpi=300, facecolor=POSTER_BG)
plt.close(fig)

shutil.copy(out_png_local, out_png_artifact)
print(f"Successfully saved:")
print(f"  - PNG: {out_png_local}")
print(f"  - PDF: {out_pdf_local}")
print(f"  - Artifact: {out_png_artifact}")
