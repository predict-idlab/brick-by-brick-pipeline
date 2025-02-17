import pandas as pd
import pickle
from tqdm import tqdm
from glob import glob
from feature_extract import extract_features
from utils import top_feats, y_columns
import numpy as np
import argparse
from datetime import datetime
import joblib


## extract features

####### MAIN #########

def main():
    parser = argparse.ArgumentParser(description="Process pickle files in a folder.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Path to the folder containing .pkl files")
    args = parser.parse_args()

    all_features = []
    for file in tqdm(glob(args.directory)):  ##
           with open(file, 'rb') as f:
                  name = file.split('/')[-1]
                  time_series_data = pickle.load(f)
                  df = pd.DataFrame(time_series_data)
                  df.columns = ["time", "value"]
                  start_date = pd.Timestamp('2023-01-01')
                  df['time'] = start_date + df['time']

                  features = {}
                  for c in ['full', 'interval_5T', 'interval_1H', 'interval_1D', 'interval_1W']:
                         features.update(extract_features(c, df))

                  features['filename'] = name
                  all_features.append(features)

    df = pd.DataFrame(all_features)
    filenames = df['filename']

    df = df[top_feats]
    df.fillna(0, inplace=True)
    df = df.clip(lower=-1e10, upper=1e10)

    print(df.columns)
    exit(0)

    model = joblib.load("model.pkl")

    y_pred_values = model.predict(df)

    ## TRANSFORM MULTICLASS BACK TO MULTILABELS
    y_pred = pd.DataFrame(np.zeros((df.shape[0], len(y_columns))), columns=y_columns)
    for r in range(len(y_pred_values)):
             for value in y_pred_values[r].split('$'):
                 if value != '':
                     y_pred.loc[r, value] = 1

    df_submission = y_pred#sum(preds_frames) / len(preds_frames)
    df_submission.insert(0, 'filename', filenames)  # Insert 'filename' as the first column.

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{timestamp}.csv.gz"
    df_submission.to_csv(filename, index=False, compression='gzip')

if __name__ == "__main__":
    main()
