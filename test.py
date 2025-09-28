import cv2
import os

file_path = r'C:\VS Code\retinopathy_project\data\images\0a4e1a29ffff.png'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

img = cv2.imread(file_path)

#if img is None:
    #raise ValueError(f"Failed to load image: {file_path}")

img = img[:, :, ::-1]
import matplotlib.pyplot as plt
plt.imshow(img)
plt.axis('off')
plt.show()