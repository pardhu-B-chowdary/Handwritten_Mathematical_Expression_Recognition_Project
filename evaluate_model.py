import os
import pandas as pd
import re

from nltk.translate.bleu_score import sentence_bleu
from jiwer import cer

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from model_inference import run_inference




# -----------------------------
# CONFIG
# -----------------------------

IMAGE_FOLDER = "./dataset/images"
CSV_FILE = "./dataset/labels.csv"


def normalize_latex(text):

    text = text.strip()

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces around operators
    text = re.sub(r"\s*([\+\-\*/=\^\(\)\{\}])\s*", r"\1", text)

    return text
    
# -----------------------------
# LOAD CSV
# -----------------------------

df = pd.read_csv(CSV_FILE)

results = []

exact_matches = 0
total = 0

all_true_tokens = []
all_pred_tokens = []


# -----------------------------
# EVALUATION LOOP
# -----------------------------

for index, row in df.iterrows():

    filename = row["image"]
    ground_truth = str(row["latex"])

    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    print(f"Processing: {filename}")

    prediction = run_inference(image_path)

    prediction_norm = normalize_latex(prediction)
    ground_truth_norm = normalize_latex(ground_truth)

    # -----------------------------
    # EXACT MATCH
    # -----------------------------

    exact = int(prediction_norm == ground_truth_norm)

    if exact:
        exact_matches += 1

    total += 1


    # -----------------------------
    # BLEU SCORE
    # -----------------------------

    bleu = sentence_bleu(
    [ground_truth_norm.split()],
    prediction_norm.split()
    )

    # -----------------------------
    # CHARACTER ERROR RATE
    # -----------------------------

    error_rate = cer(
        ground_truth_norm,
        prediction_norm
    )


    # -----------------------------
    # TOKEN METRICS
    # -----------------------------

    gt_tokens = ground_truth.split()
    pred_tokens = prediction.split()

    min_len = min(
        len(gt_tokens),
        len(pred_tokens)
    )

    gt_tokens = gt_tokens[:min_len]
    pred_tokens = pred_tokens[:min_len]

    all_true_tokens.extend(gt_tokens)
    all_pred_tokens.extend(pred_tokens)


    # -----------------------------
    # SAVE RESULTS
    # -----------------------------

    results.append({
        "Image": filename,
        "Ground Truth": ground_truth,
        "Prediction": prediction,
        "Exact Match": exact,
        "BLEU": bleu,
        "CER": error_rate
    })


# -----------------------------
# FINAL METRICS
# -----------------------------

accuracy = exact_matches / total

precision = precision_score(
    all_true_tokens,
    all_pred_tokens,
    average="micro",
    zero_division=0
)

recall = recall_score(
    all_true_tokens,
    all_pred_tokens,
    average="micro",
    zero_division=0
)

f1 = f1_score(
    all_true_tokens,
    all_pred_tokens,
    average="micro",
    zero_division=0
)

avg_bleu = sum(r["BLEU"] for r in results) / total

avg_cer = sum(r["CER"] for r in results) / total


# -----------------------------
# PRINT RESULTS
# -----------------------------

print("\n========== FINAL RESULTS ==========")

print(f"Exact Match Accuracy : {accuracy:.4f}")
print(f"Precision            : {precision:.4f}")
print(f"Recall               : {recall:.4f}")
print(f"F1 Score             : {f1:.4f}")
print(f"Average BLEU Score   : {avg_bleu:.4f}")
print(f"Average CER          : {avg_cer:.4f}")


# -----------------------------
# SAVE CSV OUTPUT
# -----------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "evaluation_results.csv",
    index=False
)

print("\nResults saved to evaluation_results.csv")