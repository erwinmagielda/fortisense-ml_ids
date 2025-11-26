import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======== CONFIG ========
sns.set(style="whitegrid")
plt.rcParams["figure.autolayout"] = True   # auto-fit for any monitor
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain.csv")
TEST_PATH = os.path.join(DATA_DIR, "KDDTest.csv")

# ======== LOAD DATA ========
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

# ======== Helper: maximise matplotlib window ========
def maximise_window():
    """Maximise matplotlib window across different backends."""
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state("zoomed")            # Windows TkAgg
    except:
        try:
            mng.window.showMaximized()        # Qt backend
        except:
            pass

# ============================================================
# PART 1 — REQUIRED EDA OUTPUTS ONLY
# ============================================================

# === 1. Dataset shapes ===
print("=== Dataset Shapes ===")
print("Training dataset shape:", df_train.shape)
print("Testing dataset shape: ", df_test.shape)
print()

# === 2. Summary Statistics ===
print("=== Summary Statistics (Training Set) ===")
# Exclude non-numerical columns
df_numeric = df_train.drop(columns=["attack_type"])
print(df_numeric.describe())
print()

# === 3. Percentage distribution of normal vs attack ===
print("=== Percentage Distribution of Normal vs Attack (Training Set) ===")
label_dist = df_train["label"].value_counts(normalize=True) * 100
print(label_dist.rename("percentage"))
print()

# === 4. Bar chart — normal vs attack (train + test) ===
label_counts_train = df_train["label"].value_counts().reset_index()
label_counts_train.columns = ["label", "count"]
label_counts_train["dataset"] = "Train"

label_counts_test = df_test["label"].value_counts().reset_index()
label_counts_test.columns = ["label", "count"]
label_counts_test["dataset"] = "Test"

combined_counts = pd.concat([label_counts_train, label_counts_test])

plt.figure()
sns.barplot(
    data=combined_counts,
    x="dataset",
    y="count",
    hue="label",
)
plt.title("Normal vs Attack Distribution (Train vs Test)")

maximise_window()
plt.show()

# === 5. Correlation heatmap (numeric only) ===
plt.figure()
corr_matrix = df_numeric.corr()
sns.heatmap(corr_matrix, cmap="coolwarm", linewidths=0.3)
plt.title("Feature Correlation Heatmap (Numeric Features Only)")

maximise_window()
plt.show()

# === 6. Full attack type distribution (required by tutor) ===
attack_counts = df_train["attack_type"].value_counts()

plt.figure()
sns.barplot(
    x=attack_counts.index,
    y=attack_counts.values,
    color="steelblue"
)
plt.xticks(rotation=90)
plt.title("Attack Type Distribution (Full Dataset)")
plt.ylabel("Count")
plt.xlabel("Attack Type")

maximise_window()
plt.show()

print("=== EDA completed successfully ===")
