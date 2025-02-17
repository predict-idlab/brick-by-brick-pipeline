# Inference Module for Time-Series Prediction

## Overview
The main.py script processes time-series data stored in `.pkl` files, extracts relevant features, and uses a pre-trained machine learning model to make predictions. The results are saved in a timestamped `.csv.gz` file.

## Dependencies
Ensure you have the following Python libraries installed:

```sh
numpy==1.23.5
pandas==2.2.3
pyentrp==1.0.0
scipy==1.15.1
tqdm==4.65.0
```
You can use the requirements.txt file within this folder.
Additionally, ensure you have `feature_extract.py` and `utils.py` in the same directory, as they provide necessary functions.

## Usage

### Running the script
You can run the script using the command:

```sh
python main.py -d "/path/to/pickle/files"
```

### Arguments
- `-d, --directory`: The path to the folder containing the time series `.pkl` files.

## How It Works
1. **Reads Pickle Files**: The script loads all `.pkl` files from the specified directory.
2. **Extracts Features**: Uses the `extract_features` function from `feature_extract.py` to process time-series data.
3. **Prepares Data for Prediction**:
   - Cleans the extracted features.
   - Clips values within a defined range.
4. **Loads Pre-Trained Model**: The script loads `model.pkl` to predict labels.
5. **Formats Predictions**:
   - Transforms multiclass predictions back to multi-label format.
   - Inserts filenames for reference.
6. **Saves Output**: The predictions are saved as a timestamped `.csv.gz` file.

## Output
The script generates an output file with the following format:

```
submission_YYYYMMDD_HHMMSS.csv.gz
```

This file contains predictions with filenames included as the first column.

## Notes
- Ensure `model.pkl` is available in the working directory.
- Missing values are replaced with `0`.
- The prediction results are stored in a compressed CSV format.