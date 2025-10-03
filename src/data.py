import os, cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

#jk = None

def read_image(path):
    img = cv2.imread(path)[:,:,::-1]

    #plt.imshow(img)
    #plt.show()
    #global jk
    #jk = img

    return img

def basic_preprocess(img, size=512):
    h,w,_ = img.shape
    mn = min(h,w)
    cy, cx = h//2, w//2
    img = img[cy-mn//2:cy+mn//2, cx-mn//2:cx+mn//2]
    img = cv2.resize(img, (size,size))
    return img

def get_transforms(train=False, size=512):
    if train:
        return A.Compose([
            A.RandomResizedCrop(size, size, scale=(0.9,1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=25, p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([A.Resize(size,size), A.Normalize(), ToTensorV2()])

class FundusDataset(Dataset):
    def __init__(self, csv_file, images_dir, transforms):
        csv_file = r"C:\VS Code\retinopathy_project\data\labels.csv"
        images_dir = r"C:\VS Code\retinopathy_project\data\images"

        self.df = pd.read_csv(csv_file , names = ['image','label'])
        self.images_dir = images_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pat = os.path.join(self.images_dir, row['image'])
        path = (f"{pat}.png")

        #print("Reading:", path)
        img = read_image(path)

        #if img is None:
            #raise FileNotFoundError(f"Image not found at {path}")

        img = basic_preprocess(img)
        img = self.transforms(image=img)['image']
        label = int(row['label'])
        return img, label
      

#Code below is only for testing
if __name__ == "__main__":
    transforms = get_transforms(train=False)
    dataset = FundusDataset("csv_file", "images_dir", transforms)
    print(f"Number of samples: {len(dataset)}")
    img, label = dataset[1]
    print(f"Image shape: {img.shape}, Label: {label}")
    
    #if jk is None:
        #print("Can't print")
    #else:
        #plt.imshow(jk)
        #plt.show()