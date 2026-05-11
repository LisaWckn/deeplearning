import pandas as pd

# CSV einlesen
train_df = pd.read_csv("Training_set.csv")

# Anzahl pro Label zählen
counts = train_df["label"].value_counts()

print("Training_set")
print(counts)

missing = train_df["label"].isna().sum()
print("\nOhne Label:", missing, "\n")

# CSV einlesen
test_df = pd.read_csv("Testing_set.csv")

merged_df = test_df.merge(train_df, on="filename", how="left")

# Anzahl pro Label zählen
counts = merged_df["label"].value_counts()

print("Testing_set")
print(counts)
