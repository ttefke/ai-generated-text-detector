import pickle

# Check results with GBM
def check_gbm_data(data):
    with open("./data/gbm", "rb") as f:
        gbm = pickle.load(f)
        
    # Predict
    return gbm.predict_proba(data)