import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from collections import Counter
import math
import scipy.stats as ss
from scipy.stats import kendalltau, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from ..misc.preprocessing import test_cat
from itertools import product
from fairnesstest.settings import BASE_DIR


def conditional_entropy(x, y):
  y_counter = Counter(y)
  xy_counter = Counter(list(zip(x,y)))
  total_occurrences = sum(y_counter.values())
  entropy = 0
  for xy in xy_counter.keys():
      p_xy = xy_counter[xy] / total_occurrences
      p_y = y_counter[xy[1]] / total_occurrences
      entropy += p_xy * math.log(p_y/p_xy)
  return entropy

def u(x, y):
  s_xy = conditional_entropy(x,y)
  x_counter = Counter(x)
  total_occurrences = sum(x_counter.values())
  p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
  s_x = ss.entropy(p_x)
  if s_x == 0:
      return 1
  else:
      return (s_x - s_xy) / s_x

def theils_u(x, y):
  x_counter = Counter(x)
  y_counter = Counter(y)
  total_occurrences_x = sum(x_counter.values())
  total_occurrences_y = sum(y_counter.values())
  p_x = list(map(lambda n: n/total_occurrences_x, x_counter.values()))
  s_x = ss.entropy(p_x)
  p_y = list(map(lambda n: n/total_occurrences_y, y_counter.values()))
  s_y = ss.entropy(p_y)
  return ((s_x * u(x, y)) + (s_y * u(y, x))) / (s_x + s_y)

def eta(x, y):
  group_var = np.array([x[y == val].mean() for val in set(y)]).std() ** 2
  overall_var = x.std() ** 2
  return np.math.sqrt(group_var / overall_var)

def cramers_v(data):
  chi2 = ss.chi2_contingency(data)[0]
  n = data.sum().sum()
  return np.sqrt(chi2 / (n*(min(data.shape)-1)))

def pointbiserial(x, y):
  s = np.std(x)
  n = x.size
  m0 = x[y == 0].mean()
  m1 = x[y == 1].mean()
  n0 = x[y == 0].size
  n1 = x[y == 1].size
  return ((m1-m0)/s) * np.sqrt((n1*n0)/(n*(n-1)))

def calculate_corr(x, y, corrtype = 'kendall'):
  if corrtype == 'pearson':
    corr, _ = pearsonr(x, y)
  elif corrtype == 'kendall':
    corr, _ = kendalltau(x, y)
  elif corrtype == 'eta':
    corr = eta(x, y)
  elif corrtype == 'pointbiserial':
    corr = pointbiserial(x, y)
  elif corrtype == 'cramer':
    corr = cramers_v(pd.crosstab(index = x, columns = y))
  elif corrtype == 'theilsu':
    corr = theils_u(x, y)
  return corr

def checkCorr(columnName_protectedVar, columName_targetVar, df, corrTh = 0.3,
              cont_cat = 'eta', cat_cat = 'kendall', cont_cont = 'kendall'):
    '''
    Checks correlation between protected attribute and target variable and also
    gives a list of possible proxies for the protected attribute.

    Parameters:
    -	columnName_protectedVar: the column name of a sensitive variable (first par)
    -	columName_targetVar: the column name of the target variable (second par)
    -	df: the dataframe for the data set (third var)
    -	corrTh: the min threshold for correlation

    Returns:
    A tuple of the form (bool, list) which has the following.
    -	if the given sensitive variable is correlated with the target variable
    -	the list of other columns the sensitive variable is correlated with (list of proxy variables)

    '''

    if test_cat(df[columnName_protectedVar].values) and test_cat(df[columName_targetVar].values):
        corr = calculate_corr(df.loc[:, columnName_protectedVar].values,
                          df.loc[:, columName_targetVar].values,
                          corrtype = cat_cat)

    elif test_cat(df[columnName_protectedVar].values) and not test_cat(df[columName_targetVar].values):
        corr = calculate_corr(df.loc[:, columnName_protectedVar].values,
                          df.loc[:, columName_targetVar].values,
                          corrtype = cont_cat)

    elif not test_cat(df[columnName_protectedVar].values) and test_cat(df[columName_targetVar].values):
        corr = calculate_corr(df.loc[:, columnName_protectedVar].values,
                          df.loc[:, columName_targetVar].values,
                          corrtype = cont_cat)

    else:
        corr = calculate_corr(df.loc[:, columnName_protectedVar].values,
                          df.loc[:, columName_targetVar].values,
                          corrtype = cont_cont)

    corrbool = abs(corr) >= corrTh

    corrs = []
    proxies = []
    statements = []
    for col in df.columns:
        if col != columnName_protectedVar and col != columName_targetVar:
            if test_cat(df[columnName_protectedVar].values) and test_cat(df[col].values):
                c = calculate_corr(df[columnName_protectedVar].values, df[col].values, corrtype = cat_cat)
            elif test_cat(df[columnName_protectedVar].values) and not test_cat(df[col].values):
                c = calculate_corr(df[columnName_protectedVar].values, df[col].values, corrtype = cont_cat)
            elif not test_cat(df[columnName_protectedVar].values) and test_cat(df[col].values):
                c = calculate_corr(df[columnName_protectedVar].values, df[col].values, corrtype = cont_cat)
            else:
                c = calculate_corr(df[columnName_protectedVar].values, df[col].values, corrtype = cont_cont)

            corrs.append(c)
            if abs(c) >= corrTh:
                proxies.append(col)


    if corrbool:
        statements.append(f"The protected variable '{columnName_protectedVar}' is correlated with the target variable '{columName_targetVar}' with a {'positive' if corr > 0 else 'negative'} correlation strength of {abs(corr)}.")
    else:
        statements.append(f"The protected variable '{columnName_protectedVar}' is not correlated with the target variable '{columName_targetVar}' with a correlation coefficient of at least {corrTh}. The correlation coefficient is {round(corr, 3)} < {corrTh}.")

    if len(proxies) > 0:
        statements.append(f"The list of proxy variables for the protected attribute '{columnName_protectedVar}' is {proxies}.")
    else:
        statements.append(f"There are no proxies for '{columnName_protectedVar}'. Highest correlation coefficient with other variables is {round(max(corrs), 2)}.")

    return (statements, corrbool, proxies)

def plotCorrMap(ColumnNameList_protectedVar, df,
                cont_cat = 'eta', cat_cat = 'kendall', cont_cont = 'kendall'):
    '''
    Generates a correlation heatmap between protected attributes and all other features.

    Parameters:
    - ColumnNameList_protectedVar: list of columns which are regarded as protected attributes
    - df: pandas dataframe under investigation

    Returns:
    - Heatmap of correlation coefficients
    '''
    heatmap_fig, heatmap_ax = plt.subplots(constrained_layout=True)
    #heatmap_fig.set_figheight(10)
    #heatmap_fig.set_figwidth(len(ColumnNameList_protectedVar))
    #plt.figure(figsize = (20, len(ColumnNameList_protectedVar)))

    heatmap_ax.set_title('Correlation between protected attributes and other features', fontsize = 10)
    #plt.title('Correlation between protected attributes and other features', fontsize = 10);

    othercols = set(df.columns).difference(ColumnNameList_protectedVar)
    corr_df = pd.DataFrame(index = ColumnNameList_protectedVar, columns = othercols)
    
    for i, (row, col) in enumerate(product(ColumnNameList_protectedVar, othercols)):
        
      test_cat_row_vals, test_cat_col_vals = test_cat(df[row].values), test_cat(df[col].values)

      if test_cat_row_vals and test_cat_col_vals:
          corr_type = cat_cat
          #corr_df.loc[row, col] = calculate_corr(df[row].values, df[col].values, corrtype = cat_cat)
      elif test_cat_row_vals and not test_cat_col_vals:
          corr_type = cont_cat
          #corr_df.loc[row, col] = calculate_corr(df[row].values, df[col].values, corrtype = cont_cat)
      elif not test_cat_row_vals and test_cat_col_vals:
          corr_type = cont_cat
          #corr_df.loc[row, col] = calculate_corr(df[row].values, df[col].values, corrtype = cont_cat)
      else:
          corr_type = cont_cont
          #corr_df.loc[row, col] = calculate_corr(df[row].values, df[col].values, corrtype = cont_cont)

      corr_df.loc[row, col] = calculate_corr(df[row].values, df[col].values, corrtype = corr_type)

    #Filter out columns with values -corr_threshold < corr < corr_threshold
    
    corr_threshold = 0.3 #Set threshold here
    corr_df.mask(abs(corr_df)<corr_threshold, inplace=True) #Changing all values to NaN
    total_valid_corrs = corr_df.count().sum()

    num_strong_corrs = 3

    if total_valid_corrs > num_strong_corrs:
      corr_df.dropna(how = 'all', axis = 0, inplace = True) #dropping rows with all NaNs
      corr_df.dropna(how = 'all', axis = 1, inplace = True) #dropping columns with all NaNs
      corr_df.fillna(0, inplace = True)
      heatmap_statement = 'Correlation values are shown in heatmap.'

    elif total_valid_corrs >0 and total_valid_corrs<=num_strong_corrs:
      corr_df.dropna(how = 'all', axis = 0, inplace = True) #dropping rows with all NaNs
      corr_df.dropna(how = 'all', axis = 1, inplace = True) #dropping columns with all NaNs

      heatmap_statement = [] #list with heatmap related name and values
      for prot_var in corr_df.index:
        prot_var_vals = corr_df.loc[prot_var] > corr_threshold
        other_cols_list = list(prot_var_vals[prot_var_vals==True].index)

        for col in other_cols_list:
          heatmap_statement.append(f'Correlation value between {prot_var} & {col} is {corr_df.loc[prot_var, col]}.')
      #corr_df.fillna(0, inplace = True)
      corr_df = None
    else:
      heatmap_statement = 'No strong correlation found between any variable.'
      corr_df = None

    #sys.stdout.write('\r')
    #sys.stdout.write(f'Building correlation matrix: {round(((i + 1) / (len(ColumnNameList_protectedVar) * len(othercols)) * 100), 3)} %')
    #sys.stdout.flush()

    #heatmap_ax.set_yticks(va = 'center')
    #cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    #sns.heatmap(corr_df.astype(float), cmap = cmap)

    print('corr_analysis---------\n', corr_df)
    if not isinstance(corr_df,type(None)):
      _ = sns.heatmap(ax = heatmap_ax, data=corr_df.astype(float), center=0, cmap='RdYlBu', 
      linewidths=0.5, linecolor='white', vmin=-1, vmax=1)
      plt.savefig(BASE_DIR/'structureddata/static/structureddata/exclusion_bias_images/correlation_heatmap.png', dpi = 300)
      
      return True, None #If heatmap is made return True and No statement
    else:
      return False, heatmap_statement #If heatmap is not made then a False and statements