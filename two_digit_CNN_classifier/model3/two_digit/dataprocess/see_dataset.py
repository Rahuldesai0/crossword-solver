import pandas as pd
from PIL import Image
import random

# Load the mixed dataset
df = pd.read_pickle('../../tmnist/2digit_hw_28x28.pkl')
df2 = pd.read_pickle('../../tmnist/2digit_typed_28x28.pkl')

print(df.columns)
print(df2.columns)
# print(len(df))
# # Separate 1-digit and 2-digit entries
# one_digit_df = df[df['label'].astype(str).str.len() == 1]
# two_digit_df = df[df['label'].astype(str).str.len() == 2]

# # Randomly pick one of each
# one_sample = one_digit_df.sample(1).iloc[0]
# two_sample = two_digit_df.sample(1).iloc[0]

# # Show both
# print("One-digit Label:", one_sample['label'])
# Image.fromarray(one_sample['image']).show()

# print("Two-digit Label:", two_sample['label'])
# Image.fromarray(two_sample['image']).show()
