from utils.structure import get_tree
import numpy as np

import pickle
import os


def save_and_load_pickle(file_path, func, *args, **kwargs):
    """
    Save the results of a function to a pickle file if it doesn't exist.
    Load the results from the pickle file if available.

    Args:
        file_path (str): Path to the pickle file.
        func (callable): The function whose results need to be saved/loaded.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function or the loaded pickle data.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    result = func(*args, **kwargs)
    with open(file_path, 'wb') as f:
        pickle.dump(result, f)

    return result
# Recursive function to extract active categories based on the tree
def extract_categories(row, tree):
    """
    Extracts categories based on active (value=1) keys in the row and a given tree structure.
    This version preserves the order of categories based on the tree depth.

    Args:
        row (dict-like): Row with one-hot encoded categories as keys.
        tree (dict): Tree structure mapping parent categories to child categories.

    Returns:
        list: List of active categories in the hierarchy, maintaining order.
    """
    result = []  # Use a list to maintain order


    # Function to avoid adding duplicates while preserving order
    def add_category(category):
        if category not in result:
            result.append(category)

    # Iterate through the tree
    for key, children in tree.items():
        # Check if the current key is active
        if key in row and row[key] == 1:
            add_category(key)

        # If children is a dictionary (subtree), recurse
        if isinstance(children, dict):
            result.extend(extract_categories(row, children))

        # If children is a list, check each for being active
        elif isinstance(children, list):
            for child in children:
                if child in row and row[child] == 1:
                    add_category(child)

    return result

def get_y_restult(y):
    tree = get_tree()
    y_result = np.array([extract_categories(row, tree) for _, row in y.iterrows()], dtype=object)
    return y_result

def labels_to_tree(y):
    if len(y) == 1:
        return get_y_restult(y)
    else:
        return save_and_load_pickle("utils/results.pkl", get_y_restult, y)