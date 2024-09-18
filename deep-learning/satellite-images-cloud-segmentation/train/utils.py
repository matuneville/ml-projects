import torch

def compute_accuracy(preds, targets):
    # Assuming preds and targets have shape [batch_size, height, width]
    preds = preds.sigmoid().round() # Apply sigmoid and round to get binary predictions
    correct = (preds == targets).float()  # Element-wise comparison
    acc = correct.sum() / targets.numel()  # Compute accuracy
    return acc.item()  # Return as a float
