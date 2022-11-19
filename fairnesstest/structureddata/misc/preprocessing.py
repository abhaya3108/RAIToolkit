import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def test_cat(arr):
  return (len(set(arr)) / len(arr)) < 0.00001

def preprocess_data(data, skipcols = None, to_drop = None, frac = 0.6, **kwargs):
  '''
  Function that does basic data preprocessing. Does the following:
  - Remove columns NULL values more than 'frac' (since they do not contribute to the analysis much)
  - Impute remaining NULL values with column mean or column mode
  - One-Hot encoding of categorical columns
  - Scaling of numerical columns

  Parameters:
  - data = pd.DataFrame under investigation
  - skipcols = list of column names that should be skipped from preprocessing treatment (optional)
  - to_drop = list of column names that need to be dropped (optional)
  - frac = fraction of tolerance for NULL values in a column (default 0.7)

  Returns:
  A pd.DataFrame that has preprocessed data

  '''

  if to_drop:
    data = data.drop(to_drop, axis = 1)

  if skipcols:
    data_skipped = data.loc[:, skipcols]
    data = data.drop(skipcols, axis = 1)
  else:
    data_skipped = pd.DataFrame()

  data = data.loc[:, data.isna().sum() / data.shape[0] <= frac]
  num_cols = data.select_dtypes(exclude=['object', 'category']).columns #data.describe().columns
  cat_cols = data.select_dtypes(include=['object', 'category']).columns #list(set(data.columns).difference(num_cols))

  if len(num_cols) > 0:
    data.loc[:, num_cols] = data.loc[:, num_cols].fillna(data.loc[:, num_cols].mean())
    
    # Transferred the min max scaler function to run only when numerical variables are present. - Shagun 28/06/2022
    sc = MinMaxScaler()
    data.loc[:, num_cols] = sc.fit_transform(data.loc[:, num_cols])
    
  if len(cat_cols) > 0:
    data.loc[:, cat_cols] = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent').fit_transform(data.loc[:, cat_cols])
    data.loc[:, cat_cols] = SimpleImputer(missing_values = None, strategy = 'most_frequent').fit_transform(data.loc[:, cat_cols])

  data = pd.get_dummies(data, **kwargs)

  data = pd.concat([data, data_skipped], axis = 1)

  return data
