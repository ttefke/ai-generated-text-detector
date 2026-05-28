# Define function to obtain the data
# (returns human texts, ai texts and labels)

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import numpy as np
import pandas as pd
import pickle
import torch
from torch.nn.functional import normalize

from .directory import *

nltk.download('punkt_tab')

# Returns labelled data in random order combined with labels or human + ai + sorted labels
def get_dataset(path, randomize=True):
    # Load data from file
    data = pd.read_csv(path)
    human_texts = data.loc[data["generated"] == 0, "text"]
    ai_texts = data.loc[data["generated"] == 1, "text"]
    
    # Put texts into single array
    texts = np.concatenate([human_texts, ai_texts])

    # Create labels
    labels = np.concatenate([
        np.zeros(len(human_texts), dtype=int),
        np.ones(len(ai_texts), dtype=int)
    ])
    
    if randomize:    
        # Combine text and labels into matrix
        combined = np.array([texts, labels])
        
        # Transform matrix so that each text is in an array with its label
        combined = combined.T
        
        # Shuffle
        np.random.shuffle(combined)
        
        # Transform back
        combined = combined.T
        
        # Return data with label
        return combined[0].tolist(), combined[1].tolist()
    else:
        return human_texts.tolist(), ai_texts.tolist(), labels
    
def get_dataset_train(randomize=True):
    return get_dataset(get_path_train(), randomize=randomize)
    
def get_dataset_test(randomize=True):
    return get_dataset(get_path_test(), randomize=randomize)

def get_dataset_validation(randomize=True):
    return get_dataset(get_path_validation(), randomize=randomize)
    
# Get dataset but in the form of sbert embeddings instead
# of plain text
def get_dataset_sbert(path):
    # Load data from file
    with open(path, "rb") as f:
        human_data, ai_data = pickle.load(f)

    X = torch.cat([human_data, ai_data], dim=0)

    # Create labels
    y = torch.cat([
        torch.zeros(len(human_data), dtype=int),
        torch.ones(len(ai_data), dtype=int)
    ])

    return (X, y)

def get_dataset_sbert_train():
    return get_dataset_sbert(get_sbert_path_train())

def get_dataset_sbert_validation():
    return get_dataset_sbert(get_sbert_path_validate())

# Get path of combined dataset
def get_path_train():
    return "./data/datasets/combined_train.csv"

# Get path of combined validation dataset
def get_path_test():
    return "./data/datasets/combined_test.csv"

# Get path of combined validation dataset
def get_path_validation():
    return "./data/datasets/combined_validation.csv"

def tokenize_text(texts):
    params = PunktParameters()
    params.sent_end_chars=('.',)
    params.abbrev_types = {'e.g', 'i.e', 'e.g.', 'i.e.'} 
    # Normalieze  'e.g.', 'et al.'

    texts = texts.replace("e. g.", "_e_g_")
    texts = texts.replace("et al.", "_et_al_")
    tokenizer = PunktSentenceTokenizer(params)
    sentences = tokenizer.tokenize(texts)

    # 'unnormalize'
    sentences = [s.replace("_e_g_", "e.g.") for s in sentences]
    sentences = [s.replace("_et_al_", "et al.") for s in sentences]

    return sentences

def clean_token(token):
    # Remove all texts shorter than 10 characters (likely not a full sencence)
    token = [item for item in token if len(item) >= 10]

    return token

def load_encoder():
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("./data/models/ai_human_minilm_retrained")
    return encoder

def encode_texts(flat_texts, device, encoder=None):
    if encoder == None:
        encoder = load_encoder()
    data = encoder.encode(flat_texts, device=device, convert_to_tensor=True)
    data = normalize(data, p=2, dim=1)
    data = data.float()
    return data