import pandas as pd
from PIL import Image
import random 
import numpy as np 

def crop_digit(img):
    # Find rows/cols where there's any non-zero pixel
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin:rmax+1, cmin:cmax+1]

def create_2digit_dataset(df, examples=50000, pad=2, outer_pad=5):
    data = []
    for _ in range(examples):
        choice1 = df.iloc[random.randrange(len(df))]
        choice2 = df.iloc[random.randrange(len(df))]
        
        img1 = crop_digit(choice1['image'].astype('uint8'))
        img2 = crop_digit(choice2['image'].astype('uint8'))
        
        # Add black padding between digits
        spacer = np.zeros((max(img1.shape[0], img2.shape[0]), pad), dtype=np.uint8)
        
        # Match heights before concatenating
        h = max(img1.shape[0], img2.shape[0])
        img1 = np.pad(img1, ((0, h - img1.shape[0]), (0, 0)), constant_values=0)
        img2 = np.pad(img2, ((0, h - img2.shape[0]), (0, 0)), constant_values=0)
        
        combined_img = np.hstack((img1, spacer, img2))

        combined_img = np.pad(
            combined_img,
            ((outer_pad, outer_pad), (outer_pad, outer_pad)),
            constant_values=0
        )
        
        label = int(f"{choice1['label']}{choice2['label']}")
        data.append({'label': label, 'image': combined_img})
    
    return pd.DataFrame(data)

import pandas as pd
import numpy as np
from PIL import Image

df = pd.read_csv('./tmnist/TMNIST_Data.csv')
df = df.drop(['names'], axis=1)

labels = df['labels']
pixels = df.drop(columns='labels').to_numpy() 

images = [row.reshape(28, 28).astype(np.uint8) for row in pixels]

new_df = pd.DataFrame({
    'label': labels,
    'image': images
})

new_df.to_pickle('tmnist/1digit_mnist.pkl')
img = Image.fromarray(new_df.iloc[0]['image'])
label = new_df.iloc[0]['label']
print(label)
img.show()
