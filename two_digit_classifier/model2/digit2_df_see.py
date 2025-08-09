import pandas as pd
import numpy as np
from PIL import Image

df = pd.read_pickle('tmnist/2digit_mnist.pkl')
import random

# Pick a random row
row = df.iloc[random.randrange(len(df))]

# Show label
print("Label:", row['label'])

# Convert numpy array to PIL Image and display
img = Image.fromarray(row['image'].astype('uint8'), mode='L')
img.show()
