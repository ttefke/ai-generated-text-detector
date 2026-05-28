import cupy as cp
import pickle
from .directory import get_svm_path


# Check data with SVM
def check_svm(data):
    # Load SVM
    with open(get_svm_path(), "rb") as f:
        (scaler, svm) = pickle.load(f)

    # Prepare data for CUDA SVM
    data_np = data.cpu().numpy().astype("float32")
    data_cp = cp.asarray(data_np)
    data_cp = scaler.transform(data_cp)

    # Predict
    y_pred = svm.predict_proba(data_cp)

    # Return results
    svm_scores = y_pred.get().tolist()
    
    del scaler, svm, data_cp, y_pred
    return svm_scores