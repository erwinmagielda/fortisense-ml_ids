import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sets a clean and consistent visual style for all plots.
sns.set_theme(style="whitegrid")

# Determines the base directory so the script can reliably locate the dataset folder.
project_root_directory = os.path.dirname(os.path.dirname(__file__))
dataset_directory = os.path.join(project_root_directory, "data")

training_dataset_path = os.path.join(dataset_directory, "KDDTrain.csv")
testing_dataset_path = os.path.join(dataset_directory, "KDDTest.csv")

# ============================================================
# PART 1 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Loads both the training and testing datasets into memory.
training_dataframe = pd.read_csv(training_dataset_path)
testing_dataframe = pd.read_csv(testing_dataset_path)

# ------------------------------------------------------------
# Dataset Shapes
# ------------------------------------------------------------
print("=== Dataset Shapes ===")
print("Training dataset shape:", training_dataframe.shape)
print("Testing dataset shape: ", testing_dataframe.shape)
print()

# ------------------------------------------------------------
# Summary Statistics
# ------------------------------------------------------------
# Removes the attack_type column because it contains strings and should not be included in numeric statistics.
numeric_training_dataframe = training_dataframe.drop(columns=["attack_type"])

print("=== Summary Statistics (Training Set) ===")
print(numeric_training_dataframe.describe())
print()

# ------------------------------------------------------------
# Label Distribution (Normal vs Attack)
# ------------------------------------------------------------
label_percentage_distribution = (
    training_dataframe["label"].value_counts(normalize=True) * 100
)

print("=== Percentage Distribution of Normal vs Attack (Training Set) ===")
print(label_percentage_distribution.rename("percentage"))
print()

# ------------------------------------------------------------
# Bar Chart – Normal vs Attack (Train vs Test)
# ------------------------------------------------------------

training_label_counts = training_dataframe["label"].value_counts().reset_index()
training_label_counts.columns = ["label", "count"]
training_label_counts["dataset"] = "Training"

testing_label_counts = testing_dataframe["label"].value_counts().reset_index()
testing_label_counts.columns = ["label", "count"]
testing_label_counts["dataset"] = "Testing"

combined_label_counts = pd.concat([training_label_counts, testing_label_counts])

plt.figure(figsize=(8, 6))
sns.barplot(
    data=combined_label_counts,
    x="dataset",
    y="count",
    hue="label",
)
plt.title("Normal vs Attack Distribution in Training and Testing Datasets")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Correlation Heatmap (Numeric Features Only)
# ------------------------------------------------------------
feature_correlation_matrix = numeric_training_dataframe.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    feature_correlation_matrix,
    cmap="coolwarm",
    linewidths=0.3,
)
plt.title("Correlation Heatmap of Numerical Features (Training Dataset)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Attack Type Distribution (All Types)
# ------------------------------------------------------------

attack_type_frequencies = (
    training_dataframe["attack_type"].value_counts()
)

plt.figure(figsize=(14, 6))
sns.barplot(
    x=attack_type_frequencies.index,
    y=attack_type_frequencies.values,
    color="steelblue",
)
plt.xticks(rotation=90)
plt.title("Distribution of All Attack Types (Training Dataset)")
plt.xlabel("Attack Type")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("=== EDA completed successfully ===")
