from matplotlib.image import imread
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

heights = []
widths = []

target_folder = '/Users/apple/PycharmProjects/AutomaticScreenshotClassification/image_files/training_set/comments/'

for image_file in os.listdir(target_folder):
    try:
        height, width, channel = imread(target_folder + image_file).shape
        heights.append(height)
        widths.append(width)
    except:
        pass

sns.scatterplot(heights, widths)
plt.show()

np.mean(heights)  # 343.41946902654865
np.mean(widths)  # 743.3982300884956

image_shape = (50, 100, 3)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_generator = ImageDataGenerator(width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.1,
                                     zoom_range=0.1,
                                     fill_mode='nearest')

sample_img = imread('image_files/training_set/comments/Screen Shot 2020-02-17 at 11.28.46 AM.png')
plt.imshow(sample_img)
plt.show()

plt.imshow(image_generator.random_transform(sample_img))
plt.show()

'''
structure of training/testing image files
image_files
    train
        comments
        non_comments
    test
        comments
        non_comments
'''

train_path = 'image_files/training_set'
test_path = 'image_files/test_set'

image_generator.flow_from_directory(train_path)  # Found 721 images belonging to 2 classes.
image_generator.flow_from_directory(test_path)  # Found 150 images belonging to 2 classes.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2, 2), input_shape=(50, 100, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(2, 2), input_shape=(50, 100, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(2, 2), input_shape=(50, 100, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3)

batch_size = 16

train_image_generator = image_generator.flow_from_directory(train_path,
                                                            target_size=(50, 100),
                                                            color_mode='rgb',
                                                            batch_size=batch_size,
                                                            class_mode='binary')

test_image_generator = image_generator.flow_from_directory(test_path,
                                                           target_size=(50, 100),
                                                           color_mode='rgb',
                                                           batch_size=batch_size,
                                                           class_mode='binary',
                                                           shuffle=False)

model.fit_generator(train_image_generator,
                    epochs=25,
                    validation_data=test_image_generator,
                    callbacks=[early_stop])

model.evaluate_generator(test_image_generator)  # loss, acc: [0.8078150361776352, 0.8466667]

from tensorflow.keras.preprocessing import image

test_img = '/Users/apple/Desktop/Screen Shot 2020-02-18 at 6.01.53 PM.png'
my_image = image.load_img(test_img, target_size=(50, 100, 3))

my_image_array = image.img_to_array(my_image)
my_image_array = np.expand_dims(my_image_array, axis=0)

model.predict(my_image_array)

results = []
for image_file in os.listdir(target_folder):
    try:
        test_img = target_folder + image_file
        my_image = image.load_img(test_img, target_size=(50, 100, 3))

        my_image_array = image.img_to_array(my_image)
        my_image_array = np.expand_dims(my_image_array, axis=0)

        results.append(model.predict(my_image_array))
    except:
        pass

results = np.array(results)
results.mean()  # 0.014322343 for comments, 0.7579729 for non_comments
results.max()  # 0.036334578 for comments

model.save('screenshots_model.h5')