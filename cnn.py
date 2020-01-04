# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# classifier.fit_generator(training_set,
#                          samples_per_epoch = 8000,
#                          nb_epoch = 1,
#                          validation_data = test_set,
#                          verbose=1,
#                          validation_steps=2000//32)

# classifier.fit_generator(training_set,
#                          samples_per_epoch = 400,
#                          nb_epoch = 5,
#                          validation_data = test_set,
#                          verbose=1,
#                          validation_steps=100//32)

classifier.fit_generator(training_set,
                         samples_per_epoch = 400,
                         nb_epoch = 25,
                         validation_data = test_set,
                         # nb_val_samples = 100,
                         validation_steps=100//32)

from pathlib import Path

model_structure = classifier.to_json()
f = Path('screenshot_model_structure2.json')
f.write_text(model_structure)

classifier.save_weights('screenshot_model_weights2.h5')

from keras.preprocessing import image

# Load an image file to test, resizing it to 32x32 pixels (as required by this model)
img = image.load_img("/Users/apple/Desktop/Screen Shot 2019-09-09 at 9.16.16 PM.png", target_size=(64, 64))
# Convert the image to a numpy array
image_to_test = image.img_to_array(img)
image_to_test /= 255

import numpy as np
# Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
list_of_images = np.expand_dims(image_to_test, axis=0)


results = classifier.predict(list_of_images)
print(results)