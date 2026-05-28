# Check with DistilBERT
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import torch.nn.functional as F
from .directory import get_distilbert_directory

def check_distilbert(flat_texts, device, tokenizer=None, model=None):
    results = []
    
    # Load model
    if tokenizer is None or model is None:
        tokenizer = DistilBertTokenizerFast.from_pretrained(get_distilbert_directory())
        model = DistilBertForSequenceClassification.from_pretrained(get_distilbert_directory())
        model.to(device)
        model.eval()

    # Process longer texts in multiple batches
    batches_num = 1
    if len(flat_texts) > 125:
        batches_num = len(flat_texts) / 125
        
    flat_texts = np.array(flat_texts)
    flat_texts = np.array_split(flat_texts, batches_num)
        
    for flat_text_arr in flat_texts:
        flat_text_arr = flat_text_arr.tolist()
                
        # Compute probabilities
        inputs = tokenizer(
            flat_text_arr,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        # Get probablities
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            
        inputs.to("cpu")
            
        # Compute results
        for sent, p in zip(flat_text_arr, probs):
            results.append([p[0], p[1]])

    return results