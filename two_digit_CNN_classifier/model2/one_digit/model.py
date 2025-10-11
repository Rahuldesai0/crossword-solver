
import torch
import torch.nn as nn

class MLP16x8(nn.Module):
    def __init__(self, num_classes, input_size=(16, 8)):
        super().__init__()
        self.flatten_size = input_size[0] * input_size[1]  # 16*8 = 128
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten (batch_size, 128)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)