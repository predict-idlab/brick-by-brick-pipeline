# brick-by-brick-pipeline

All necessary files to reconstruct my solution ML pipeline to achieve a 0.5767 macro F1 score on the public test leaderboard.
We refer to the accompanying report.pdf for more details.

This repository contain:
- model.py file with the code to train the model based on extracted features. Includes also the feature selection paradigm and hyperparm optimalisation code based on a stratified k-fold approach.
- brick_feature_extractor.py contains all the code to extract all relevant time, interval and full features (which are stored in the features folder for further references)
- A dataloader.py file that combines all features together for both train and test sets
- Some utiles files relate to the brick schema and available classes
- A submission folder containing all used solution submission files related to the paper.


All code was executed using Python 3.11. </br> Necessary packages are pandas (latest), scikit-learn (latest).


This code is licensed under the Apache License 2.0. See the LICENSE file for details.