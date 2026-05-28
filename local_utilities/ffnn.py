# Define FFNN
# Check with FFNN
from .directory import *
import torch
from torch.nn import Softmax

import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self, input_dim=384, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)
        

# Check data with ffnn
def check_ffnn(data, device):
    # Load FFNN
    model = FFNN()
    model.load_state_dict(torch.load(get_ffnn_path()))
    model = model.to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = Softmax(dim=1)(logits)
        preds = torch.argmax(probs, dim=1)
        
    logits.to("cpu")
    model.to("cpu")

    # Generate result values
    results = []
    for sent, p, prob in zip(data, preds, probs):
        results.append(prob.cpu().numpy())

    return results