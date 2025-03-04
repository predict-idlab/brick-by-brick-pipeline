# brick-by-brick-pipeline

This repository was developed as part of the **Brick by Brick 2024 Challenge**, a global competition aimed at **automating building data classification** to enhance the intelligence and energy efficiency of buildings. By tackling this challenge, we contribute to the broader effort of enabling **sustainable and smart building management** through data-driven solutions.

Our machine learning pipeline achieves a **0.5767 macro F1 score** on the public test leaderboard. The approach leverages advanced **feature extraction** and **ensemble learning** to optimize performance. For a detailed explanation of the methodology, feature extraction process, and model optimization, please refer to the accompanying **report.pdf** and our published paper.

## Brick by Brick 2024 Challenge
The **Brick by Brick 2024 Challenge** is a global initiative designed to advance automation in building data classification. The challenge's primary goal is to enable **intelligent, energy-efficient buildings** by leveraging machine learning and data science methodologies. Our contribution focuses on developing a robust, high-performing model that effectively processes and classifies building data.

## Provided Solution
The solution is built around **feature extraction** and **ensemble learning**, leveraging a structured approach to feature selection and hyperparameter optimization. The pipeline follows a **stratified k-fold cross-validation** scheme to ensure robustness.

### Repository Structure:
- **`model.py`**: Contains the code to train the model based on extracted features. It also includes the feature selection paradigm and hyperparameter optimization using a **stratified k-fold approach**.
- **`brick_feature_extractor.py`**: Implements feature extraction, generating **time-based, interval-based, and full dataset features**, which are stored in the `features/` folder for reference.
- **`dataloader.py`**: Merges extracted features for both training and test datasets.
- **Utility Files**: Contains schema definitions and available classes relevant to the brick classification problem.
- **`submission/`**: Includes all solution submission files related to the paper.

## Citation
If you use this code or refer to our work, please cite the following paper:

```bibtex
@inproceedings{Steenwinckel2025,
  author    = {Bram Steenwinckel and Sofie Van Hoecke and Femke Ongenae},
  title     = {Another Brick in the Wall: Leveraging Feature Extraction and Ensemble Learning for Building Data Classification},
  booktitle = {Companion Proceedings of the ACM Web Conference 2025 (WWW Companion '25)},
  year      = {2025},
  publisher = {ACM},
  doi       = {10.1145/3701716.3718480},
  isbn      = {979-8-4007-1331-6/2025/04},
}
```

## Requirements
This code was executed using **Python 3.11** with the following necessary packages:
- `pandas` (latest version)
- `scikit-learn` (latest version)

## License
This code is licensed under the **Apache License 2.0**. See the LICENSE file for details.
