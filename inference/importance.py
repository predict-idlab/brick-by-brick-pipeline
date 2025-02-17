import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = joblib.load("model.pkl")

# Try to get feature names (works for some scikit-learn models)
try:
    feature_names = model.feature_names_in_
    print(feature_names)
    feature_names = ['autocorrelation', 'sampling_rate', 'event_density', 'trigger_ratio',
                     'state_length', 'largest_state', 'dominant_state',
                     '25q_time_difference',
                     '75q_time_difference',
                     'mad_time_difference',
                     'min_tim_difference', 'permutation_entropy',
                     'sample_entropy', 'rms', 'energy', 'event_density_5T', 'trigger_ratio_5T',
                     'state_length_5T', 'largest_state_5T', 'dominant_state_5T',
                     'permutation_entropy_5T', 'rms_5T', 'spectral_entropy_1H',
                     'event_density_1H', 'trigger_ratio_1H', 'state_length_1H',
                     'largest_state_1H', 'dominant_state_1H', 'permutation_entropy_1H', 'rms_1H',
                     'smallest_state_1D', 'largest_state_1D', 'dominant_state_1D', 'rms_1D',
                     'event_density_1W', 'trigger_ratio_1W', 'smallest_state_1W',
                     'largest_state_1W', 'dominant_state_1W', 'rms_1W']

    print(feature_names)
except AttributeError:
    # Use generic feature names if not stored in the model
    n_features = len(model.feature_importances_) if hasattr(model, 'feature_importances_') else len(model.coef_.ravel())
    feature_names = [f"Feature {i}" for i in range(n_features)]

# Check if the model has feature_importances_ or coef_
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
elif hasattr(model, 'coef_'):
    importances = np.abs(model.coef_.ravel())  # Take absolute value for linear models
else:
    raise ValueError("Model does not have feature_importances_ or coef_ attribute")

# Sort the features by importance
indices = np.argsort(importances)[::-1]
sorted_features = np.array(feature_names)[indices]
sorted_importances = importances[indices]

# Plot the feature importances
fig = plt.figure(figsize=(4, 7))  # Dynamic height based on number of features
plt.barh(sorted_features, sorted_importances, color='firebrick', edgecolor='darkred', height=0.75)
plt.xlabel('Feature Importance')

# Adjust layout for better readability
plt.gca().invert_yaxis()  # Most important on top
plt.xticks(fontsize=8)
plt.yticks(fontsize=9)


max_importance = np.max(sorted_importances)
tick_interval = max_importance / 10  # More ticks, closer together
# Add more space between bars
plt.grid(axis='x', linestyle='--', alpha=0.7)
for i, a in enumerate(fig.axes):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

plt.tight_layout()

# Save the figure with high resolution (300 dpi)
plt.savefig('feature_importance_high_quality.png', dpi=600)

# Show the plot
plt.show()
plt.show()
