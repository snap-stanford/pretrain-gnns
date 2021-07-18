from sklearn.metrics import confusion_matrix
import numpy as np


def logAUC():
    pass

def enrichment(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    p = np.sum(y_true == 1)
    n = np.sum(y_true == 0)
    print(f'tn:{tn} fp:{fp}, fn:{fn} tp:{tp} p:{p}, n:{n}')
    if(tp + fp) != 0:
        enr = (tp / (tp + fp)) / (p / (p + n))
    else:
        enr = np.NAN
    return enr


if __name__ == '__main__':
    y_true = np.array([1, 0, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0])
    out = enrichment(y_true, y_pred)
    print(out)
