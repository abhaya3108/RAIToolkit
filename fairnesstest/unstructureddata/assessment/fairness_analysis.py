import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp
from scipy.spatial import distance

def confusionMatrixDissimilarity(mat1, mat2, normalize = True, method = 'distance', threshold = 0.1):
  '''
  Computes dissimilarity between two confusion matrices.

  Parameters:
  - mat1, mat2: Two 2x2 numpy arrays for two confusion matrices
  - normalize: Normalize the confusion matrices (only applicable if method == 'distance', default True)
  - method: Similarity computation method, choice between 'pvalue' and 'distance' (default 'pvalue')
  - threshold: Dissimilarity threshold, beyond which matrices will be flagged as dissimilar
               For 'distance' method this is the max tolerable pairwise distance, for 'p-value' method
               this is the alpha level (default 0.05)

  Returns:
  TBD


  '''
  if method == 'distance':
    if normalize:
      mat1 = mat1 / mat1.sum()
      mat2 = mat2 / mat2.sum()
    dist = distance.euclidean(mat1.ravel(), mat2.ravel())
    return {'flag': 'Biased' if dist > threshold else 'Fair', 'dist': dist}

  elif method == 'pvalue':
    _, pvalue = ks_2samp(mat1.ravel(), mat2.ravel(), alternative = 'two-sided')
    return {'flag': 'Biased' if pvalue < threshold else 'Fair', 'pvalue': pvalue}

def conf_scores(confMat):
  tn = confMat[0, 0]
  tp = confMat[1, 1]
  fn = confMat[1, 0]
  fp = confMat[0, 1]
  return tp, tn, fp, fn

def generate_dissimilarity_report(X_test, y_test, y_pred, onehot_protected_attributes, acc_metrics = 'precision_recall', **kwargs):
  '''
  Frontend for the confusionMatrixDissimilarity function. Generates a dissimilarity report for
  all input features.

  Parameters:
  - X_test: pd.DataFrame object containing test data
  - y_test: numpy array containing test labels
  - y_pred: numpy array containing predicted labels on X_test
  - onehot_protected_attributes: list of one-hot protected attributes
  - method: 'method' of confusionMatrixDissimilarity, default 'pvalue'
  - threshold: 'threshold' of confusionMatrixDissimilarity, default 0.1

  Returns:
  A pd.DataFrame containing confusion matrix dissimilarity scores and flags for all one-hot encoded
  features. For details on the column names, see confusionMatrixDissimilarity docstring.

  '''
  rowdicts = []
  for attr in onehot_protected_attributes:
    d = {'feature': attr}
    subpop0 = (y_test[X_test[attr] == 0], y_pred[X_test[attr] == 0])
    subpop1 = (y_test[X_test[attr] == 1], y_pred[X_test[attr] == 1])
    mat0 = confusion_matrix(subpop0[0], subpop0[1])
    mat1 = confusion_matrix(subpop1[0], subpop1[1])
    print("------------------------", mat0)
    print("------------------------", mat1)
    d.update(confusionMatrixDissimilarity(mat0, mat1, **kwargs))
    tp0, tn0, fp0, fn0 = conf_scores(mat0)
    tp1, tn1, fp1, fn1 = conf_scores(mat1)
    if acc_metrics == 'precision_recall':
        prec0 = round(tp0 / (tp0 + fp0), 4)
        prec1 = round(tp1 / (tp1 + fp1), 4)
        rec0 = round(tp0 / (tp0 + fn0), 4)
        rec1 = round(tp1 / (tp1 + fn1), 4)
        d.update({'prec0, rec0': (prec0, rec0), 'prec1, rec1': (prec1, rec1)})
    elif acc_metrics == 'tpr_fpr':
        tpr0 = round(tp0 / (tp0 + fn0), 4)
        tpr1 = round(tp1 / (tp1 + fn1), 4)
        fpr0 = round(fp0 / (fp0 + tn0), 4)
        fpr1 = round(fp1 / (fp1 + tn1), 4)
        d.update({'tpr0, fpr0': (tpr0, fpr0), 'tpr1, fpr1': (tpr1, fpr1)})

    rowdicts.append(d)

  return pd.DataFrame(rowdicts)

def falsePositiveRateParity(confusionMatrix):
  tp, tn, fp, fn = conf_scores(confusionMatrix)
  return fp / (tn + fp)

def falseDiscoveryRateParity(confusionMatrix):
  tp, tn, fp, fn = conf_scores(confusionMatrix)
  return fp / (tp + fp)

def equalizedOdds(confusionMatrix):
  tp, tn, fp, fn = conf_scores(confusionMatrix)
  fpr = fp / (tn + fp)
  tpr = tp / (fn + tp)
  return 1 - abs(tpr - fpr)

def PPVParity(confusionMatrix):
  tp, tn, fp, fn = conf_scores(confusionMatrix)
  return tp / (tp + fp)

def NPVParity(confusionMatrix):
  tp, tn, fp, fn = conf_scores(confusionMatrix)
  return tn / (fn + tn)

def equalOpportunity(confusionMatrix):
  tp, tn, fp, fn = conf_scores(confusionMatrix)
  return tp / (fn + fp)

def falseNegativeRateParity(confusionMatrix):
  tp, tn, fp, fn = conf_scores(confusionMatrix)
  return fn / (tp + fn)

def falseOmissionRateParity(confusionMatrix):
  tp, tn, fp, fn = conf_scores(confusionMatrix)
  return fn / (tn + fn)

def generateFairnessReport(X_test, y_test, y_pred, onehot_protected_attributes):
  rowdicts = []
  for attr in onehot_protected_attributes:
    y_test_0 = y_test[X_test[attr] == 0]
    y_test_1 = y_test[X_test[attr] == 1]
    y_pred_0 = y_pred[X_test[attr] == 0]
    y_pred_1 = y_pred[X_test[attr] == 1]
    confusionMatrix_0 = confusion_matrix(y_test_0, y_pred_0)
    confusionMatrix_1 = confusion_matrix(y_test_1, y_pred_1)
    rowdicts.append({
        'Feature': attr,
        'FPR Parity': (round(falsePositiveRateParity(confusionMatrix_0), 3), round(falsePositiveRateParity(confusionMatrix_1), 3)),
        'FNR Parity': (round(falseNegativeRateParity(confusionMatrix_0), 3), round(falseNegativeRateParity(confusionMatrix_1), 3)),
        'FDR Parity': (round(falseDiscoveryRateParity(confusionMatrix_0), 3), round(falseDiscoveryRateParity(confusionMatrix_1), 3)),
        'FOR Parity': (round(falseOmissionRateParity(confusionMatrix_0), 3), round(falseOmissionRateParity(confusionMatrix_1), 3)),
        'PPV Parity': (round(PPVParity(confusionMatrix_0), 3), round(PPVParity(confusionMatrix_1), 3)),
        'NPV Parity': (round(NPVParity(confusionMatrix_0), 3), round(NPVParity(confusionMatrix_1), 3)),
        'Equalized Odds': (round(equalizedOdds(confusionMatrix_0), 3), round(equalizedOdds(confusionMatrix_1), 3)),
        'Equal Opportunity': (round(equalOpportunity(confusionMatrix_0), 3), round(equalOpportunity(confusionMatrix_1), 3))
     })

  return pd.DataFrame(rowdicts)
