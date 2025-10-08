import pandas as pd
import numpy as np
from PIL import Image
import os
import random

def crop_digit(img_np):
    rows = np.any(img_np > 0, axis=1)
    cols = np.any(img_np > 0, axis=0)
    if not rows.any() or not cols.any():
        return img_np  # skip blank
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img_np[rmin:rmax+1, cmin:cmax+1]

def create_2digit_handwritten_dataset(csv_path, img_dir, output_pkl, limit=50000, size=28):
    df = pd.read_csv(csv_path)
    df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)

    data = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        img = Image.open(img_path).convert('L')
        img_np = np.array(img, dtype=np.uint8)

        # Crop around the digit content
        cropped = crop_digit(img_np)

        # Resize to standard 28x28
        img_resized = np.array(Image.fromarray(cropped).resize((size, size), Image.LANCZOS), dtype=np.uint8)

        label = int(f"{row['label1']}{row['label2']}")
        data.append({'label': label, 'image': img_resized})

    final_df = pd.DataFrame(data)
    final_df.to_pickle(output_pkl)
    print(f"Saved {len(final_df)} samples to {output_pkl}")

    # Show a random example
    sample = final_df.sample(1).iloc[0]
    print("Example Label:", sample['label'])
    Image.fromarray(sample['image']).show()

# Example usage
create_2digit_handwritten_dataset(
    csv_path='../../tmnist/train.csv',
    img_dir='../../tmnist/train',
    output_pkl='../../tmnist/2digit_handwritten_28x28.pkl'
)
