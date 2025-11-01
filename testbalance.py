import pandas as pd

df = pd.read_csv(r"C:\VS Code\retinopathy_project\data\combined\labels_train.csv", names=['image', 'label'])
print(df['label'].value_counts())