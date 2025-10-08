import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=100, img_size=(28,28)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)      # 1×28×28 → 6×24×24
        self.pool = nn.MaxPool2d(2)                      # 6×24×24 → 6×12×12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # 6×12×12 → 16×8×8
        self.row, self.col = img_size

        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.row, self.col)
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
