import cv2
import torch
import numpy as np
from ocr import ocr_image
import os
from skimage.morphology import skeletonize

# copied same function for debugging
def preprocess_digit(img_bin, digit_size=28, final_size=(28, 56), sharpen=True, skeletonize_digit=True, debug=False):
    img = img_bin.copy()

    if len(img.shape) == 3:  # convert BGR to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img.max() > 1 and img.max() <= 255:
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros(final_size, dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    x_off = (size - w) // 2
    y_off = (size - h) // 2
    square[y_off:y_off+h, x_off:x_off+w] = digit

    resized = cv2.resize(square, (digit_size, digit_size), interpolation=cv2.INTER_AREA)

    # Erosion to remove very small isolated pixels
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    eroded = cv2.erode(resized, kernel_erode, iterations=1)

    # Morphological closing to fill small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    morphed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_close)

    if sharpen:
        kernel_sharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]], dtype=np.float32)
        morphed = cv2.filter2D(morphed, -1, kernel_sharp)
        morphed = np.clip(morphed, 0, 255).astype(np.uint8)

    if skeletonize_digit:
        bool_img = morphed > 0
        skeleton = skeletonize(bool_img)
        morphed = (skeleton.astype(np.uint8) * 255)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        morphed = cv2.dilate(morphed, kernel_dilate, iterations=2)

    final = np.zeros(final_size, dtype=np.uint8)
    x_pad = (final_size[1] - digit_size) // 2
    final[:, x_pad:x_pad+digit_size] = morphed

    if debug:
        print(f"preprocess_digit: orig {img_bin.shape}, bbox {(w,h)}, digit_size {digit_size}, final_size {final_size}")

    return final

class DigitModel:
    def __init__(self, model_choice=2, device='cpu'):
        self.device = device
        self.model = None

        if model_choice == 1:
            from two_digit_classifier.model3.two_digit.model import CNN
            self.model = CNN(num_classes=100)
            model_path = './two_digit_classifier/model3/two_digit/model_combined.pth'
        elif model_choice == 2:
            from two_digit_classifier.model2.two_digit.model import CNN
            self.model = CNN(num_classes=100)
            model_path = './two_digit_classifier/model2/two_digit/model1.pth'
        else:
            raise ValueError("Invalid model_choice. Use 1 or 2.")

        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def predict(self, img_tensor, top_k=5):
        with torch.no_grad():
            outputs = self.model(img_tensor.to(self.device))
            _, indices = torch.topk(outputs, k=top_k, dim=1)
        return indices[0].cpu().tolist()

model_instance = DigitModel(model_choice=2)

def predict_image(img_np, min_size=10, ocr=False, save_path=None):
    img_np = preprocess_digit(img_np)
    
    if save_path:
        cv2.imwrite(save_path, img_np)
    
    if ocr:
        return img_np

    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float() / 255.0

    return model_instance.predict(img_tensor)


if __name__ == '__main__':
    img = cv2.imread("./cells/cell_0_0.png")
    digits = predict_image(img, save_path="./processed_cells/cell_0_0_processed.png")
    print("Predicted digits:", digits)

    cv2.imshow("Processed Digit", preprocess_digit(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
