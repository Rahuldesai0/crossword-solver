import torch
import torch.nn as nn
import numpy as np
from PIL import Image

def predict_image(image_path, model):
    target_h = 28
    target_w = 56

    img = Image.open(image_path).convert('L')
    img = img.resize((target_w, target_h), Image.BILINEAR)

    img_arr = np.array(img, dtype=np.float32)
    img_arr = 255.0 - img_arr  # invert colors

    img_tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred_idx = outputs.argmax(1).item()

    return pred_idx

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
    
model = LeNet(num_classes=100)
model.load_state_dict(torch.load('./model1.pth', map_location='cpu'))
model.eval()

for i in range(1, 6):
    print(predict_image(f'./test_images/test{i}.jpg', model))