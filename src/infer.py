import torch, cv2, matplotlib.pyplot as plt
from src.model import create_model
from src.data import basic_preprocess, get_transforms

model = create_model(n_classes=5)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu', weights_only=True))
model.eval()

path = input("Enter image path: ").strip()
img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(f"Could not load image from {path}")
img = img[:, :, ::-1]

img = basic_preprocess(img, size=512)
transform = get_transforms(train=False, size=512)
tensor = transform(image=img)['image'].unsqueeze(0)

with torch.no_grad():
    out = model(tensor)
    pred = out.argmax(dim=1).item()

print("Predicted grade:", pred)
plt.imshow(img); 
plt.title(f"Pred: {pred}"); 
plt.axis('off')
