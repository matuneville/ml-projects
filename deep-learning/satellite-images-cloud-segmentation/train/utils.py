import torch

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()