from utils.brick_utils import labels_to_tree
from utils.data import unique_labels
import pandas as pd
import numpy as np

def load_training_data():
    one = pd.read_csv('features/extended_train_features.csv', index_col=0)
    two = pd.read_csv('features/extended_train_5T_features.csv', index_col=0).drop(list(unique_labels), axis=1)
    two = two.rename(columns={c: c + '_5T' for c in two.columns if c not in ['filename']})
    three = pd.read_csv('features/extended_train_1H_features.csv', index_col=0).drop(list(unique_labels), axis=1)
    three = three.rename(columns={c: c + '_1H' for c in three.columns if c not in ['filename']})
    four = pd.read_csv('features/extended_train_1D_features.csv', index_col=0).drop(list(unique_labels), axis=1)
    four = four.rename(columns={c: c + '_1D' for c in four.columns if c not in ['filename']})
    five = pd.read_csv('features/extended_train_1W_features.csv', index_col=0).drop(list(unique_labels), axis=1)
    five = five.rename(columns={c: c + '_1W' for c in five.columns if c not in ['filename']})
    six = pd.read_csv('features/interval_features.csv', index_col=0).drop(list(unique_labels), axis=1)
    six = six.rename(columns={c: c + '_special' for c in six.columns if c not in ['filename']})

    final_df = one.merge(two, on="filename").merge(three, on="filename").merge(four, on="filename").merge(five, on="filename").merge(six, on="filename")#pd.read_csv('extended_train_5T_features.csv', index_col=0)
    X = final_df.drop(['filename']+list(unique_labels), axis=1)
    ###
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    X = X.clip(lower=-1e10, upper=1e10)

    y = (final_df[list(unique_labels)]==1).astype(int)
    y_result = labels_to_tree(y)

    groups = final_df[['Alarm', 'Command', 'Parameter', 'Status']].idxmax(axis=1)
    return X, y, y_result, groups

def load_test_data():
    one = pd.read_csv('features/extended_test_features.csv', index_col=0)
    two = pd.read_csv('extended_test_5T_features.csv', index_col=0)
    two = two.rename(columns={c: c + '_5T' for c in two.columns if c not in ['filename']})
    three = pd.read_csv('features/extended_test_1H_features.csv', index_col=0)
    three = three.rename(columns={c: c + '_1H' for c in three.columns if c not in ['filename']})
    four = pd.read_csv('features/extended_test_1D_features.csv', index_col=0)
    four = four.rename(columns={c: c + '_1D' for c in four.columns if c not in ['filename']})
    five = pd.read_csv('features/extended_test_1W_features.csv', index_col=0)
    five = five.rename(columns={c: c + '_1W' for c in five.columns if c not in ['filename']})
    six = pd.read_csv('features/interval_test_features.csv', index_col=0)
    six = six.rename(columns={c: c + '_special' for c in six.columns if c not in ['filename']})

    final_df = one.merge(two, on="filename").merge(three, on="filename").merge(four, on="filename").merge(five,
                                                                                                          on="filename").merge(six, on="filename")  # pd.read_csv('extended_train_5T_features.csv', index_col=0)
    filenames = final_df['filename']
    X_test = final_df.drop(['filename'], axis=1)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(0, inplace=True)
    X_test = X_test.clip(lower=-1e10, upper=1e10)
    return X_test, filenames