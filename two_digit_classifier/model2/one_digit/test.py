import torch
import numpy as np
from PIL import Image
from model import MLP16x8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
model = MLP16x8(num_classes, input_size=(16, 8)).to(device)
model.load_state_dict(torch.load('./model1_1digit.pth', map_location=device))
model.eval()

def predict_digit_top3(image_path, model, target_size=(16, 8)):
    target_h, target_w = target_size
    img = Image.open(image_path).convert('L')
    print(img.size)
    img = img.resize((target_w, target_h), Image.BILINEAR)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        values, indices = torch.topk(output, k=3, dim=1)
    return indices[0].cpu().tolist(), values[0].cpu().tolist()


for i in range(1, 100):
    try:
        image_path = f"./test_images/test{i}.jpg"
        top3_indices, top3_values = predict_digit_top3(image_path, model)
        print(top3_indices)
    except: 
        break
