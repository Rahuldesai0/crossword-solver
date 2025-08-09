from .model import model
import numpy as np
from PIL import Image
import os

# Load model weights
model.load_weights(r'two_digit_classifier\two-digit-custom-model-test2-with-batchnormalization.h5')

def prep_data_keras_single(img_array):
    # Reshape to (1, 64, 64, 1), convert to float32 and normalize
    input_img = img_array.reshape(1, 64, 64, 1).astype('float32') / 255.0
    return input_img

i = 1
def load_and_inspect_image(image_path):
    global i
    # Load image
    img = Image.open(image_path).convert('L').resize((64, 64))
    img_array = np.array(img)
    
    # Print image details for debugging
    print(f"Image loaded from: {image_path}")
    print(f"Image shape: {img_array.shape}, dtype: {img_array.dtype}")
    print(f"Pixel min/max: {img_array.min()}/{img_array.max()}")
    
    # Save the processed image for visual inspection (optional)
    debug_path = os.path.join(os.path.dirname(image_path), f"debug_input_{i}.png")
    Image.fromarray(img_array).save(debug_path)
    print(f"Saved debug image to: {debug_path}")
    i += 1
    
    return img_array


def predict_digits(img_array):
    """Use this function. Pass in a numpy array to this image."""
    """You will need to invert the image, as the model works for white handwriting on black"""
    input_img = prep_data_keras_single(img_array)
    predictions = model.predict(input_img)
    
    digit1 = np.argmax(predictions[0][0])
    digit2 = np.argmax(predictions[1][0])
    
    return digit1, digit2

if __name__ == "__main__":
    test_images = [
        r'two_digit_classifier/test.jpg',
        r'two_digit_classifier/test2.jpg',
        r'two_digit_classifier/test3.jpg',
    ]
    
    for path in test_images:
        img_array = load_and_inspect_image(path)
        digit1, digit2 = predict_digits(img_array)
        print(f"Predicted digits for {os.path.basename(path)}: {digit1}, {digit2}\n")
