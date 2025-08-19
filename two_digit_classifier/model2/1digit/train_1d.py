import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

from model import MLP16x8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DigitDataset(Dataset):
    def __init__(self, df, target_size=(16, 8)):
        target_h, target_w = target_size
        processed_images = []

        for img in df['image']:
            img = img.astype(np.uint8)                 
            pil_img = Image.fromarray(img)            
            pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
            processed_images.append(np.array(pil_img, dtype=np.uint8))

        self.images = np.stack(processed_images)
        self.labels = df['label'].to_numpy()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert to tensor and add channel dimension
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label
    
df = pd.read_pickle('tmnist/1digit_mnist.pkl')
classes = sorted(df['label'].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
df['label'] = df['label'].map(class_to_idx)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_loader = DataLoader(DigitDataset(train_df), batch_size=64, shuffle=True)
test_loader = DataLoader(DigitDataset(test_df), batch_size=64)

model = MLP16x8(len(classes), (16, 8)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Epoch {epoch+1}: Test Acc = {correct/total:.4f}")

torch.save(model.state_dict(), 'model1_1digit.pth')