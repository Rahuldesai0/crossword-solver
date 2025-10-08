import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from model import CNN

def predict_image(image_path, model):
    target_h = 28
    target_w = 56

    img = Image.open(image_path).convert('L')
    img = img.resize((target_w, target_h), Image.BILINEAR)

    img_arr = np.array(img, dtype=np.float32)
    img_arr = 255.0 - img_arr 

    img_tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        # pred_idx = outputs.argmax(1).item()
        values, indices = torch.topk(outputs, k=3, dim=1)

    return indices[0].cpu().tolist()

model = CNN(num_classes=100)
model.load_state_dict(torch.load('./model1.pth', map_location='cpu'))
model.eval()

for i in range(1, 100):
    try:
        print(predict_image(f'./test_images/test{i}.jpg', model))
    except Exception:
        break