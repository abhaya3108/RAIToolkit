import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
import mpld3

def confusionMatrixDissimilarity(mat1, mat2, method, threshold, normalize = True):
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
      if not np.all(mat1==0):
        mat1 = mat1 / mat1.sum()
      if not np.all(mat2==0):
        mat2 = mat2 / mat2.sum()
    dist = distance.euclidean(mat1.ravel(), mat2.ravel())
    return {'Flag': 'Biased' if dist > threshold else 'Fair', 'Difference in Model Performance': dist}

  elif method == 'pvalue':
    _, pvalue = ks_2samp(mat1.ravel(), mat2.ravel(), alternative = 'two-sided')
    return {'Flag': 'Biased' if pvalue < threshold else 'Fair', 'KS Test P-Value': pvalue}

def conf_scores(confMat):
  tn = confMat[0, 0]
  tp = confMat[1, 1]
  fn = confMat[1, 0]
  fp = confMat[0, 1]
  return tp, tn, fp, fn

def generate_dissimilarity_report(X_test, y_test, y_pred, method, threshold, acc_metrics = 'precision_recall', **kwargs):
  '''
  Frontend for the confusionMatrixDissimilarity function. Generates a dissimilarity report for
  all input features.

  Parameters:
  - X_test: pd.DataFrame object containing test data
  - y_test: numpy array containing test labels
  - y_pred: numpy array containing predicted labels on X_test
  - method: 'method' of confusionMatrixDissimilarity, default 'pvalue'
  - threshold: 'threshold' of confusionMatrixDissimilarity, default 0.1

  Returns:
  A pd.DataFrame containing confusion matrix dissimilarity scores and flags for all one-hot encoded
  features. For details on the column names, see confusionMatrixDissimilarity docstring.

  '''
  rowdicts = []
  for attr in X_test.columns:
    d = {'feature': attr}
    subpop0 = (y_test[X_test[attr] == 0], y_pred[X_test[attr] == 0])
    subpop1 = (y_test[X_test[attr] == 1], y_pred[X_test[attr] == 1])
    mat0 = confusion_matrix(subpop0[0], subpop0[1], labels = [0,1])
    mat1 = confusion_matrix(subpop1[0], subpop1[1], labels = [0,1])
    print("------------------------", mat0)
    print("------------------------", mat1)
    d.update(confusionMatrixDissimilarity(mat0, mat1, method, threshold, **kwargs))
    tp0, tn0, fp0, fn0 = conf_scores(mat0)
    tp1, tn1, fp1, fn1 = conf_scores(mat1)
    if acc_metrics == 'precision_recall':
        # prec0 = round(tp0 / (tp0 + fp0), 4)
        prec1 = round(tp1 / (tp1 + fp1), 4)
        # rec0 = round(tp0 / (tp0 + fn0), 4)
        rec1 = round(tp1 / (tp1 + fn1), 4)
        d.update({'Precision, Recall': (prec1, rec1)})
    elif acc_metrics == 'tpr_fpr':
        # tpr0 = round(tp0 / (tp0 + fn0), 4)
        tpr1 = round(tp1 / (tp1 + fn1), 4)
        # fpr0 = round(fp0 / (fp0 + tn0), 4)
        fpr1 = round(fp1 / (fp1 + tn1), 4)
        d.update({'TPR, FPR': (tpr1, fpr1)})

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

def plot_cf_matrix(cf_matrix):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix[1].flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         cf_matrix[1].flatten()/np.sum(cf_matrix[1])]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    plt.title(cf_matrix[0], fontsize = 15)
    return sns.heatmap(cf_matrix[1], annot=labels, fmt='', cmap='Purples', cbar=False, xticklabels=False, yticklabels=False) 

def generateFairnessReport(X_test, y_test, y_pred, feats_dict):
  rowdicts = []
  figs = []
  for feats in feats_dict:
    cf_matrices = []
    for attr in feats_dict[feats]:
      # for attr in onehot_protected_attributes:
      # y_test_0 = y_test[X_test[attr] == 0]
      y_test_1 = y_test[X_test[attr] == 1]
      # y_pred_0 = y_pred[X_test[attr] == 0]
      y_pred_1 = y_pred[X_test[attr] == 1]
      # confusionMatrix_0 = confusion_matrix(y_test_0, y_pred_0, labels = [0,1])
      confusionMatrix_1 = confusion_matrix(y_test_1, y_pred_1, labels = [0,1])
      rowdicts.append({
          'Feature': attr,
          'FPR Parity': round(falsePositiveRateParity(confusionMatrix_1), 3),
          'FNR Parity': round(falseNegativeRateParity(confusionMatrix_1), 3),
          'FDR Parity': round(falseDiscoveryRateParity(confusionMatrix_1), 3),
          'FOR Parity': round(falseOmissionRateParity(confusionMatrix_1), 3),
          'PPV Parity': round(PPVParity(confusionMatrix_1), 3),
          'NPV Parity': round(NPVParity(confusionMatrix_1), 3),
          'Equalized Odds': round(equalizedOdds(confusionMatrix_1), 3),
          'Equal Opportunity': round(equalOpportunity(confusionMatrix_1), 3)
      })
      cf_matrices.append([attr, confusionMatrix_1])
    
    print(f"Plotting confusion matrix for {feats} feature.")
    #Checking the distribution of other features
    i = 1
    plt.figure()
    fig, ax = plt.subplots(figsize=(12,12))
#     for cf in [confusionMatrix_0, confusionMatrix_1]:
    number_of_subplots=len(cf_matrices)

    if number_of_subplots <= 10:
      for i,v in enumerate(range(number_of_subplots)):
          
          plt.subplot(number_of_subplots,3,v+1)
          plot_cf_matrix(cf_matrices[v])
          i += 1
      plt.tight_layout()
      figs.append(mpld3.fig_to_html(fig, no_extras = True))
      
  return pd.DataFrame(rowdicts), figs