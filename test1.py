import cv2
import matplotlib.pyplot as plt
import pandas as pd
#import os

df = pd.read_csv("data/labels.csv")
print(df.head())
img_name = df.iloc[0]['id_code'] 

#print(img_name) 
#file_path = f"data/images/{img_name}"
#print(file_path)

label = df.iloc[0]['diagnosis'] 
#print(os.path.exists(file_path))
     
img = cv2.imread(f"data/images/{img_name}.png")[:,:,::-1]
plt.imshow(img)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()