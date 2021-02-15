import pandas as pd
import numpy as np
import scipy
from sklearn.metrics import mean_squared_error, recall_score, precision_score, f1_score

def load_dataset(path):
    output = pd.read_csv(path, index_col='Id')
    return output

def measure_performance(y_pred, y_test):
    # The mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    y_pred_rounded = np.round(y_pred).astype(int)
    y_pred_clipped = np.clip(y_pred_rounded, 1, 5)
    # Recall
    print('Recall:')
    print(recall_score(y_test.values, y_pred_clipped, labels=[1,2,3,4,5], average=None))
    # Precision
    print('Precision:')
    print(precision_score(y_test.values, y_pred_clipped, labels=[1,2,3,4,5], average=None))
    print(precision_score(y_test.values, y_pred_clipped, labels=[1,2,3,4,5], average='weighted'))
    # F1 score
    print('F1 score:')
    print(f1_score(y_test.values, y_pred_clipped, labels=[1,2,3,4,5], average=None))
