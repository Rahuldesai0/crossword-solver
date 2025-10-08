import pandas as pd
import numpy as np
from PIL import Image
import random

def crop_digit(img):
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin:rmax+1, cmin:cmax+1]

def create_mixed_dataset(df, examples=50000, pad=2, outer_pad=2, size=28):
    data = []
    for _ in range(examples):
        # Randomly pick 1-digit or 2-digit
        if random.random() < 0.5:
            choice = df.iloc[random.randrange(len(df))]
            img = crop_digit(choice['image'].astype('uint8'))
            label = int(choice['label'])
        else:
            choice1 = df.iloc[random.randrange(len(df))]
            choice2 = df.iloc[random.randrange(len(df))]

            img1 = crop_digit(choice1['image'].astype('uint8'))
            img2 = crop_digit(choice2['image'].astype('uint8'))

            spacer = np.zeros((max(img1.shape[0], img2.shape[0]), pad), dtype=np.uint8)
            h = max(img1.shape[0], img2.shape[0])
            img1 = np.pad(img1, ((0, h - img1.shape[0]), (0, 0)), constant_values=0)
            img2 = np.pad(img2, ((0, h - img2.shape[0]), (0, 0)), constant_values=0)
            
            combined = np.hstack((img1, spacer, img2))
            label = int(f"{choice1['label']}{choice2['label']}")
            img = combined

        img = np.pad(img, ((outer_pad, outer_pad), (outer_pad, outer_pad)), constant_values=0)

        # Resize or pad to exactly 28x28
        h, w = img.shape
        if h > size or w > size:
            img = np.array(Image.fromarray(img).resize((size, size), Image.LANCZOS))
        else:
            pad_h = (size - h) // 2
            pad_w = (size - w) // 2
            img = np.pad(img, ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)), constant_values=0)

        data.append({'label': label, 'image': img})

    return pd.DataFrame(data)

# Load single-digit TMNIST dataset
df = pd.read_csv('../../tmnist/TMNIST_Data.csv')
df = df.drop(['names'], axis=1)

labels = df['labels']
pixels = df.drop(columns='labels').to_numpy() 
images = [row.reshape(28, 28).astype(np.uint8) for row in pixels]

one_digit_df = pd.DataFrame({
    'label': labels,
    'image': images
})

# Create mixed dataset (1-digit + 2-digit)
mixed_df = create_mixed_dataset(one_digit_df, examples=50000)
mixed_df.to_pickle('../tmnist/mixed_28x28.pkl')

# Show one random example
sample = mixed_df.sample(1).iloc[0]
print("Example Label:", sample['label'])
Image.fromarray(sample['image']).show()
