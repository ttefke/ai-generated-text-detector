import os

# Create directory if nonexistant
def create_dir_if_nonexistant(path):
    if not os.path.isdir(path):
        os.makedirs(path)

# Get directory that stores Distilbert model
def get_distilbert_directory():
    directory = "./data/models/distilbert_sentences"
    create_dir_if_nonexistant(directory)
    return directory

# Get directory that stores Roberta model
def get_roberta_directory():
    directory = "./data/models/roberta_sentences"
    create_dir_if_nonexistant(directory)
    return directory

# Get directory that stores the retrained minilm
def get_minilm_directory():
    directory = "./data/models/ai_human_minilm_retrained"
    create_dir_if_nonexistant(directory)
    return directory

# Get FFNN path
def get_ffnn_path():
    return "./data/ffnn.pth"

# Get SVM path
def get_svm_path():
    return "./data/svm"

# Get path of sbert embeddings
def get_sbert_path_train():
    return "./data/sbert_embeddings_train"

def get_sbert_path_validate():
    return "./data/sbert_embeddings_validate"

def get_sbert_path_test():
    return "./data/sbert_embeddings_test"