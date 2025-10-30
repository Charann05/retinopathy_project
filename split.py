import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\VS Code\retinopathy_project\data\combined\labels_combined.csv", skiprows = 1, names=["code_id","diagnosis"])
print(df.head)
print(df["diagnosis"].value_counts())

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["diagnosis"], random_state=42)
train_df.to_csv(r"C:\VS Code\retinopathy_project\data\combined\labels_train.csv", index=False, header=False)
val_df.to_csv(r"C:\VS Code\retinopathy_project\data\combined\labels_val.csv", index=False, header=False)
print("Split complete. Files saved as labels_train.csv and labels_val.csv")