import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DigitDataset(Dataset):
    def __init__(self, df):
        target_h = 28
        target_w = 56
        
        processed_images = []
        for img in df['image']:
            # Ensure numpy array is uint8 or float32
            img = img.astype(np.uint8)
            
            # Convert to PIL for resizing
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
            
            processed_images.append(np.array(pil_img))
        
        self.images = np.stack(processed_images)
        self.labels = df['label'].to_numpy()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label
    
class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 56)  # change if your input size differs
            dummy = self.pool(torch.relu(self.conv1(dummy)))
            dummy = self.pool(torch.relu(self.conv2(dummy)))
            flatten_size = dummy.numel()

        self.fc1 = nn.Linear(flatten_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

df = pd.read_pickle('tmnist/2digit_mnist.pkl')
classes = sorted(df['label'].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
df['label'] = df['label'].map(class_to_idx)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_loader = DataLoader(DigitDataset(train_df), batch_size=64, shuffle=True)
test_loader = DataLoader(DigitDataset(test_df), batch_size=64)

model = LeNet(len(classes)).to(device)
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

torch.save(model.state_dict(), 'model1.pth')