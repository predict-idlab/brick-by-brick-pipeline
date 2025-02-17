from dataloader import load_training_data, load_test_data
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import pickle

X, y, y_result, groups = load_training_data()
# TRANSFORM MULTILABEL TO MULTICLASS SETTING
y_result_class = np.array(['$'.join(x) for x in y_result])
class_names = list(set(y_result_class))
print(len(class_names))

y_result_uniques = list(set([x for xs in y_result for x in xs]))
print(y_result_uniques)

# Initialize Stratified K-Fold with 5 splits
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
accuracy_scores = []
y_true = []  # List to store true labels for all folds
y_pred_all = []  # List to store predicted labels for all folds

models = []

# Perform Stratified K-Fold cross-validation
# Top performing features after selection
top_feats = ['autocorrelation', 'sampling_rate', 'event_density', 'trigger_ratio',
       'state_length', 'largest_state', 'dominant_state',
       '25th Percentile of Time Diffs (seconds)',
       '75th Percentile of Time Diffs (seconds)',
       'Median Absolute Deviation (MAD) of Time Diffs (seconds)',
       'Minimum Time Difference (seconds)', 'permutation_entropy',
       'sample_entropy', 'rms', 'energy', 'event_density_5T',
       'trigger_ratio_5T', 'state_length_5T', 'largest_state_5T',
       'dominant_state_5T', 'permutation_entropy_5T', 'rms_5T',
       'spectral_entropy_1H', 'event_density_1H', 'trigger_ratio_1H',
       'state_length_1H', 'largest_state_1H', 'dominant_state_1H',
       'permutation_entropy_1H', 'rms_1H', 'smallest_state_1D',
       'largest_state_1D', 'dominant_state_1D', 'rms_1D', 'event_density_1W',
       'trigger_ratio_1W', 'smallest_state_1W', 'largest_state_1W',
       'dominant_state_1W', 'rms_1W']

best_params = {'max_depth': 30, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 4,
                      'n_estimators': 314,
                      'criterion': 'entropy'}

config = 'TEST'
search_top_features = False
use_top_features = True
tune_hyperparams = False

if config=='TRAIN':
    for train_index, test_index in kf.split(X, y_result_class):
        # Split data
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]

        ## Use top performing features if setting is true
        if use_top_features:
            X_train = X_train[top_feats]
            X_test = X_test[top_feats]
        print(len(X_train), len(X_test))

        y_train, y_test = y_result_class[train_index], y.iloc[test_index,:]

        param_dist = {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(10, 100),
            'min_samples_leaf': [2],
            'min_samples_split': [4],
            'criterion': ['entropy'],
            'class_weight': [None, 'balanced'],
            'max_features': [None]
        }

        #### Use this model option to search for best parameters
        if tune_hyperparams:
            model = RandomizedSearchCV(estimator=ExtraTreesClassifier(), param_distributions=param_dist, n_iter=30, cv=3, scoring='f1_macro', verbose=3, n_jobs=-1, random_state=42)
            print("Best parameters:", model.best_params_)
            params = model.best_params_
        else:
            params = best_params # , 'class_weight': calculate_class_weights(y_result_class)}

        model = ExtraTreesClassifier(**params)
        model.fit(X_train, y_train)

        if search_top_features:
            selector = SelectFromModel(model, threshold="mean", max_features=40)  # Select features with importance >= mean importance
            feature_idx = selector.get_support()
            selected_features = X.columns[feature_idx]
            X_train_selected = X_train[selected_features]#selector.transform(X_train)

            print(selected_features)
            X_test_selected = X_test[selected_features]

            model.fit(X_train_selected, y_train)
            y_pred_values = model.predict(X_test_selected)
        else:
            y_pred_values = model.predict(X_test)

        # Make predictions on the test set both model and postprocesing
        y_pred = pd.DataFrame(np.zeros((X_test.shape[0], y_test.shape[1])), columns=y.columns)
        for r in range(len(y_pred_values)):
             for value in (y_pred_values[r].split('$')):
                 if value != '':
                     y_pred.loc[r, value] = 1

        y_pred = y_pred.fillna(0)

        y_pred = y_pred_values


        print(")--", f1_score(y_test, y_pred, average="macro"))

        f1_scores.append(f1_score(y_test, y_pred, average="macro"))
        accuracy_scores.append(accuracy_score(y_test, y_pred))

    # Print the mean accuracy and mean micro F1 score across all folds
    mean_accuracy = np.mean(accuracy_scores)
    mean_f1 = np.mean(f1_scores)
    print(f"\nMean Accuracy across all 5 folds: {mean_accuracy:.4f}")
    print(f"Mean Micro F1 Score across all 5 folds: {mean_f1:.4f}")

else:
    model = ExtraTreesClassifier(**best_params)
    if use_top_features:
        X_train_selected = X[top_feats]
    else:
        X_train_selected = X

    model.fit(X_train_selected, y_result_class)

    #import joblib
    # save
    #joblib.dump(model, "model.pkl", compress=3)

    ## load test data
    x_test, filenames = load_test_data()
    y_pred_values = model.predict(x_test[top_feats])

    ## TRANSFORM MULTICLASS BACK TO MULTILABELS
    y_pred = pd.DataFrame(np.zeros((x_test.shape[0], y.shape[1])), columns=y.columns)
    for r in range(len(y_pred_values)):
             for value in y_pred_values[r].split('$'):
                 if value != '':
                     y_pred.loc[r, value] = 1

    df_submission = y_pred#sum(preds_frames) / len(preds_frames)
    df_submission.insert(0, 'filename', filenames)  # Insert 'filename' as the first column.

    # -----------------------------------------------------------
    # SAVE THE SUBMISSION FILE
    # -----------------------------------------------------------
    # Save as a compressed CSV (gzip) without the index column.
    # sampleweights.classweights.
    df_submission.to_csv("PUT_NICE_NAME_HERE.csv.gz", index=False, compression='gzip')