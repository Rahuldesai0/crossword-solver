import random
from .model import model

import numpy as np
from skimage.transform import resize

from keras.datasets import mnist
from keras.utils import to_categorical

import matplotlib.pyplot as plt
plt.style.use('ggplot')

mnist_img_height , mnist_img_width = 28 , 28
(x_train,y_train), (x_test, y_test) = mnist.load_data()

def build_sequence_mnist(data,labels,dataset_size,IS_TRAIN=True):
    
    # sequence data size
    seq_img_height = 64
    seq_img_width = 64
    
    seq_data = np.ndarray(shape=(dataset_size,seq_img_height,seq_img_width),
                           dtype=np.float32)
    seq_labels = [] 
    
    for i in range(0,dataset_size):
        
        #Pick a random number of digits to be in the dataset
        # num_digits = random.randint(1,2)
        
        s_indices = [random.randint(0,len(data)-1) for p in range(0,2)]

        if IS_TRAIN:
          # concatenating images and labels together
          new_image = np.hstack([x_train[index] for index in s_indices])
          new_label =  [y_train[index] for index in s_indices]
        else:

          new_image = np.hstack([x_test[index] for index in s_indices])
          new_label =  [y_test[index] for index in s_indices]
        
        
        #Resize image
        new_image = resize(new_image,(seq_img_height,seq_img_width))
        
        seq_data[i,:,:] = new_image
        seq_labels.append(tuple(new_label))
        
    
    #Return the synthetic dataset
    return seq_data,seq_labels


x_seq_train,y_seq_train = build_sequence_mnist(x_train,y_train,60000)
x_seq_test,y_seq_test = build_sequence_mnist(x_test,y_test,10000,IS_TRAIN=False)

#Converting labels to One-hot representations of shape (set_size,digits,classes)
possible_classes = 10

def convert_labels(labels):
        
    #Declare output ndarrays
    dig0_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig1_arr = np.ndarray(shape=(len(labels),possible_classes))

    
    for index,label in enumerate(labels):
        
        dig0_arr[index,:] = to_categorical(label[0],possible_classes)
        dig1_arr[index,:] = to_categorical(label[1],possible_classes)
        
    return [dig0_arr,dig1_arr]

train_labels = convert_labels(y_seq_train)
test_labels = convert_labels(y_seq_test)

def prep_data_keras(img_data):
    
    img_data = img_data.reshape(len(img_data),64,64,1)
    img_data = img_data.astype('float32')
    img_data /= 255.0
    
    return img_data

model.load_weights(r'two_digit_classifier\two-digit-custom-model-test2-with-batchnormalization.h5')

train_images = prep_data_keras(x_seq_train)
test_images = prep_data_keras(x_seq_test)
print(f"Train Image Shape: {test_images.shape}")

# scores = model.evaluate(test_images, test_labels)
# print(f"First digit accuracy: {scores[3]} , Second digit accuracy: {scores[4]}")

# scores = model.evaluate(train_images, train_labels)
# print(f"First digit accuracy: {scores[3]} , Second digit accuracy: {scores[4]}")