import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader

from src.data import FundusDataset, get_transforms
from src.model import create_model

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    val_ds = FundusDataset('data/labels_val.csv', 'data/images', get_transforms(train=False))
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = create_model(n_classes=5).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(10,6))
    indices = np.random.choice(len(val_ds), 6, replace=False)
    for ax, idx in zip(axes.flatten(), indices):
        img, label = val_ds[idx]
        model_input = img.unsqueeze(0).to(device)
        pred = model(model_input).argmax(dim=1).item()
        img_np = img.permute(1,2,0).cpu().numpy()
        ax.imshow((img_np - img_np.min()) / (img_np.max() - img_np.min()))
        ax.set_title(f"True: {label}, Pred: {pred}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()