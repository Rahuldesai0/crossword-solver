import cv2 
import torch 
import numpy as np 
from ocr import ocr_image

model_choice = 1  # set 1 or 2

model = None
try:
    if model_choice == 1:
        from two_digit_classifier.model3.two_digit.model import CNN
        model = CNN(num_classes=100)
        model_path = './two_digit_classifier/model3/two_digit/model_combined.pth'
    elif model_choice == 2:
        from two_digit_classifier.model2.two_digit.model import CNN
        model = CNN(num_classes=100)
        model_path = './two_digit_classifier/model2/two_digit/model1.pth'
    else:
        raise ValueError("Invalid model_choice. Use 1 or 2.")

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
except Exception as e:
    print("Warning: could not load model")
    print(f"Model load error: {e}")
    exit(-1)
    
def predict_image(img_np, min_size=10, ocr=False):
    if ocr:
        text, data, img = ocr_image(img_np)
        return text

    target_h = 28
    target_w = 28

    # Convert to grayscale if needed
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_np.copy()

    # Slightly blur to reduce noise
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Threshold to get binary image
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # No digit found, fallback to center crop
        h, w = img_gray.shape
        crop = img_gray[2:h-2, 2:w-2]
    else:
        # Get bounding box around all contours (union)
        x_min = min([cv2.boundingRect(c)[0] for c in contours])
        y_min = min([cv2.boundingRect(c)[1] for c in contours])
        x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
        y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])

        # Crop with a small margin (2-3 pixels)
        x_min = max(0, x_min - 2)
        y_min = max(0, y_min - 2)
        x_max = min(img_gray.shape[1], x_max + 2)
        y_max = min(img_gray.shape[0], y_max + 2)

        crop = img_gray[y_min:y_max, x_min:x_max]

    # Resize to model input
    img_resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img_arr = img_resized.astype(np.float32) / 255.0  # normalize if your model expects

    print("Image size going to model: ", img_arr.shape)
    cv2.imshow("To model", img_arr)

    img_tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        values, indices = torch.topk(outputs, k=5, dim=1)

    return indices[0].cpu().tolist()


if __name__ == '__main__':
    img = cv2.imread("./cells/cell_0_0.png")
    digit = predict_image(img)
    print(digit)
    cv2.imshow("Cell", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()