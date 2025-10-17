import numpy as np

def f1_weighted_binary(y_true, y_pred, weights):
    # y_true,y_pred: [N, T], weights: [T]
    eps = 1e-8
    tp = ((y_true==1) & (y_pred==1)).astype(float)
    fp = ((y_true==0) & (y_pred==1)).astype(float)
    fn = ((y_true==1) & (y_pred==0)).astype(float)
    w = weights.reshape(1,-1)
    p = (w*tp).sum() / ((w*(tp+fp)).sum()+eps)
    r = (w*tp).sum() / ((w*(tp+fn)).sum()+eps)
    f1 = 2*p*r/(p+r+eps)
    return float(f1)
