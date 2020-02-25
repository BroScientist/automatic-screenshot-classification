from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import shutil
import numpy as np

model = load_model('/Users/apple/PycharmProjects/AutomaticScreenshotClassification/screenshots_model.h5')

threshold = 0.15
target_folder = '/Users/apple/Desktop/'
comments_folder = '/Users/apple/Desktop/Screen Captures/Comments/'
alt_folder = '/Users/apple/Desktop/Screen Captures/Other Screenshots/'

for file in os.listdir(target_folder):
    try:
        if 'Screen Shot' in file:
            img_path = target_folder + file
            img_to_predict = image.load_img(img_path, target_size=(50, 100, 3))
            img_array = image.img_to_array(img_to_predict)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            if prediction <= threshold:
                shutil.move(img_path, comments_folder)
                print('moved image to comments folder')
            else:
                shutil.move(img_path, alt_folder)
                print('moved image to alternate folder')
    except:
        print('problem relocating image')
        pass
# target_folder = '/Users/apple/PycharmProjects/AutomaticScreenshotClassification/image_files/test_set/non_comments/'
# results = []
#
# for img_file in os.listdir(target_folder):
#     try:
#         img_path = target_folder + img_file
#         img_to_predict = image.load_img(img_path, target_size=(50, 100, 3))
#
#         img_array = image.img_to_array(img_to_predict)
#         img_array = np.expand_dims(img_array, axis=0)
#
#         prediction = model.predict(img_array)
#         results.append(prediction <= threshold)
#     except:
#         pass
#
# results = np.array(results)
# np.unique(results, return_counts=True)  # Out[37]: (array([False,  True]), array([  4, 454]))
#
#
# img_path = '/Users/apple/Desktop/Screen Shot 2020-02-18 at 11.23.04 AM.png'
# img_to_predict = image.load_img(img_path, target_size=(50, 100, 3))
#
# img_array = image.img_to_array(img_to_predict)
# img_array = np.expand_dims(img_array, axis=0)
#
# results.append(model.predict(img_array) <= threshold)