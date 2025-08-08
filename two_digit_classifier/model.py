import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop

nb_classes = 10

img_rows = 64
img_cols = 64
img_channels = 1

#number of convolution filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

#defining the input
inputs = Input(shape=(img_rows,img_cols,img_channels))

#Model taken from keras example. Worked well for a digit, dunno for multiple
cov = Convolution2D(nb_filters,kernel_size,padding='same')(inputs)
cov = Activation('relu')(cov)
cov = BatchNormalization()(cov)
cov = Convolution2D(nb_filters,kernel_size)(cov)
cov = Activation('relu')(cov)
cov = BatchNormalization()(cov)
cov = MaxPooling2D(pool_size=pool_size)(cov)
cov = Dropout(0.25)(cov)
cov_out = Flatten()(cov)
cov2 = Dense(128, activation='relu')(cov_out)
cov2 = Dropout(0.5)(cov2)


#Prediction layers
c0 = Dense(nb_classes, activation='softmax', name='digit1')(cov2)
c1 = Dense(nb_classes, activation='softmax', name='digit2')(cov2)

#Defining the model
model = Model(inputs=inputs,outputs=[c0,c1],name='custom-simple-model')

#Compiling the model
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=0.001),metrics=['accuracy', 'accuracy'])

if __name__ == '__main__':
    model.summary()
