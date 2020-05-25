import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.applications.xception import Xception
from keras.optimizers import Adam
from keras import layers
from keras import Model

pre_model = Xception(
        weights=None,
        input_shape=(150,150,3),
        include_top=False
    )

pre_weights = 'C:/Users/hp/Desktop/pneumonia/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_model.load_weights(pre_weights)

for layer in pre_model.layers:
    layer.trainable = False
    
last_layer = pre_model.get_layer('block4_pool')
last_output = last_layer.output

x = layers.AveragePooling2D(7,7)(last_output)
x = layers.Flatten()(x)
x = layers.Dense(1024,activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512,activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1,activation='sigmoid')(x)



model = Model(pre_model.input,x)
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    rotation_range=40,
    zoom_range=0.4,
    shear_range=0.2,
    width_shift_range=0.3,
    height_shift_range=0.3,
    fill_mode='nearest'
    )

train_generator = train_datagen.flow_from_directory(
    'Dataset/Train',
    batch_size=10,
    target_size=(150,150),
    class_mode='binary'
    )

test_datagen = ImageDataGenerator(rescale=1/255)

test_generator = test_datagen.flow_from_directory(
    'Dataset/Validation',
    batch_size=10,
    target_size=(150,150),
    class_mode='binary'
    )

model.fit_generator(
    train_generator,
    validation_data=test_generator,
    epochs=20
    )

print('[INFO] saving mask detector model...')
model.save('model.h5')










 