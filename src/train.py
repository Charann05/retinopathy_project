import torch, torch.nn as nn
from torch.utils.data import DataLoader
from src.data import FundusDataset, get_transforms
from src.model import create_model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = FundusDataset('C:\VS Code\retinopathy_project\data\labels.csv', 'C:\VS Code\retinopathy_project\data\images', get_transforms(train=True))
    val_ds   = FundusDataset('C:\VS Code\retinopathy_project\data\labels.csv', 'C:\VS Code\retinopathy_project\data\images', get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    model = create_model(n_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1,11):
        model.train()
        losses = []
        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch} train_loss={sum(losses)/len(losses):.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds); all_labels.extend(labels.numpy())
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch} val_acc={acc:.4f}")
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
