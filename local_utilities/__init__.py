import torch

from .dataset import clean_token, encode_texts
from .distilbert import check_distilbert
from .ffnn import check_ffnn
from .gbm import check_gbm_data
from .roberta import check_roberta
from .svm import check_svm

def evaluate(token, device):
    token = clean_token(token)
    encodings = encode_texts(token, device)

    ffnn_results = check_ffnn(encodings, device)
    torch.cuda.empty_cache()
    svm_results = check_svm(encodings)
    distilbert_results = check_distilbert(token, device)
    torch.cuda.empty_cache()
    roberta_results = check_roberta(token, device)    
    torch.cuda.empty_cache()
    
    # Apply GBM
    matrix = []
    for i in range(len(ffnn_results)):
        matrix.append([svm_results[i][1], ffnn_results[i][1],
                        distilbert_results[i][1].to("cpu"), roberta_results[i][1].to("cpu")])
    gbm_probas = check_gbm_data(matrix)
    
    return token, ffnn_results, distilbert_results, roberta_results, svm_results, gbm_probas

def generate_report(cutoff, flat_texts, ffnn_results, distilbert_results, roberta_results, svm_scores, gbm_probas, outfile):
    # Combine results
    results = []

    # Compute scores for each sentence
    for i in range(len(flat_texts)):
        result = {
            "text": flat_texts[i],
        }
        human_score = 0.0
        ai_score = 0.0
        
        table = f"AI scores:\n" # (SBERT-FFN, SBERT-SVM (bin), DistilBert, Roberta): {ai_score_s:.2f}, {score_svm}, {ai_score_d:.2f}, {ai_score_r:.2f}"

        result["ffnn"] =  {
                "human": float(ffnn_results[i][0]),
                "ai": float(ffnn_results[i][1]),
        }
        table += f"SBERT-FFNN: {result["ffnn"]["ai"]}\n"
        
        result["distilbert"] = {
            "human": float(distilbert_results[i][0]),
            "ai": float(distilbert_results[i][1]),
        }
        table += f"DistilBERT: {result["distilbert"]["ai"]}\n"

        result["roberta"] = {
            "human": float(roberta_results[i][0]),
            "ai": float(roberta_results[i][1]),
        }
        table += f"RoBERTa: {result["roberta"]["ai"]}\n"

        result["svm"] = {
            "human": float(svm_scores[i][0]),
            "ai": float(svm_scores[i][1]),
        }
        table += f"SVM: {result["svm"]["ai"]}"
        
        result["human"] = float(gbm_probas[i][0])
        result["ai"] = float(gbm_probas[i][1])
        table += f" GBM: {result["ai"]}"
        result["table"] = table

        results.append(result)
        
    # 'One-Hot'-Encoding for sentences 
    # Use a sliding window to get the score of i-1 and i+1. if both and i are over the cutoff, mark all three sentences
    ai_detections = 0
    onehot = [0] * len(results)
    for i in range(len(results)):
        # Check if we are over cutoff
        if results[i]["ai"] >= cutoff:
            # We are -> look if i-1 and i+1 are, too
            predecessor_i = i - 1
            if predecessor_i < 0:
                predecessor_i = 0
                
            successor_i = i + 1
            if successor_i > len(results) -1:
                successor_i = len(results) -1
                
            # Check predecessor and successor
            if results[predecessor_i]["ai"] >= cutoff and results[successor_i]["ai"] >= cutoff:
                # Both are over cutoff -> mark in onehot
                onehot[predecessor_i] = 1
                onehot[i] = 1
                onehot[successor_i] = 1
                ai_detections += 1

    # Generate combined report
    report = """
    <style>
    [data-tooltip]:hover::after {
    display: block;
    position: absolute;
    content: attr(data-tooltip);
    border: 2px solid black;
    background: #f2f2f2;
    padding: .5em;
    }
    </style>
    """
    
    # Generate HTML report

    ai_texts = 0
    human_texts = 0

    for i in range(len(results)):
        text = results[i]["text"]
        ai_score = (results[i]["ai"] - 0.95) * 20 # 95% cutoff
        human_score = ((results[i]["human"]) / 19) * 20 # 95% cutoff
        table = results[i]["table"]

        # Likely AI
        if onehot[i] == 1:
            ai_texts += 1
            report += f'<div style="background-color: rgba(255, 0, 0, {ai_score})" data-tooltip="{table}">{text}</div>\n'
        # Likely human
        else:
            human_texts += 1
            report += f'<div style="background-color: rgba(0, 255, 0, {human_score})" data-tooltip="{table}">{text}</div>\n'

    nr_texts = ai_texts + human_texts
    report += f'<p>Likely human: {human_texts}/{nr_texts}, likely AI: {ai_texts}/{nr_texts} ({float(ai_texts) / float(nr_texts) * 100:.2f}%)</p>'

    with open(outfile, "w") as f:
        f.write(report)
        
    return ai_detections

# Read text from a given file (path)
import re

def read_text_from_file(path):
    # Read in sentences to check

    texts = ""
    line_count = 0
    with open(path, "r") as textfile:
        for line in textfile:
            # Keep track of lines
            line_count += 1

            # Skip first 3 lines (header)
            if line_count < 3:
                continue

            # Remove unwanted sections
            if line.strip() == "References":
                break
            if line.startswith("Abstract:"):
                line = line[len("Abstract:"):]
            elif line.startswith("Abstract"):
                line = line[len("Abstract"):]
            if line.startswith("Internet of Things Projects 202"):
                continue

            # Remove em dashes at beginning (these are more likely wrongly formatted bullet lists)
            if line.startswith("– "):
                line = line[2:]

            # Merge hyphenated words at line breaks
            #line = re.sub(r'-\s*\n\s*', '', line)

            # Replace remaining newlines with space
            #line = re.sub(r'\n', ' ', line)

            # Remove reference numbers like [1], [2],
            #line = re.sub(r'\[\d+\],?', '', line)
            #line = re.sub(r'\[\d+\]', '', line)
            line = re.sub(r'\[\d+(?:,\s\d+)*],?', '', line)

            # Normalize whitespace and punctuation spacing
            line = re.sub(r'\s+', ' ', line)
            line = re.sub(r'\s+\.', '.', line)
            line = line.strip()

            texts += line + " "

    # Fix any remaining hyphenation artifacts inside the text
    texts = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', texts)

    return texts