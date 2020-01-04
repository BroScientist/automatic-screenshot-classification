from keras.models import model_from_json
from pathlib import Path
import numpy as np
from keras.preprocessing import image

#TODO retrain with more edge cases, bring acc over 95%
#TODO 
class_labels = [
    "non_comment",
    "comment"
]

f = open("/Users/apple/PycharmProjects/AutomaticScreenshotClassification/screenshot_model_structure2.json", "r")
model_structure = f.read()
model = model_from_json(model_structure)
model.load_weights("/Users/apple/PycharmProjects/AutomaticScreenshotClassification/screenshot_model_weights2.h5")

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    # Convert the image to a numpy array and scale it
    image_to_test = image.img_to_array(img)
    image_to_test /= 255

    # Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
    list_of_images = np.expand_dims(image_to_test, axis=0)


    results = model.predict(list_of_images)

    single_result = results[0]

    most_likely_class_index = int(np.argmax(single_result))
    class_likelihood = single_result[most_likely_class_index]

    print(class_likelihood)
    if class_likelihood > 0.01:
        print('image is not a comment')
        return False
    else:
        print('image is a comment')
        return True

import os
import shutil
from datetime import date
today = str(date.today())

target_folder = '/Users/apple/Desktop/Screen Captures/Comments'
alt_folder = '/Users/apple/Desktop/Screen Captures/Other Screenshots'

for f in os.listdir('/Users/apple/Desktop/'):
    try:
        if 'Screen Shot' in f:
            date_taken = f.split()[2]
            # if date_taken == today:
            image_path = '/Users/apple/Desktop/' + f
            print(image_path)
            if predict_image(image_path):
                shutil.move(image_path, target_folder)
                print('moved image to target folder')
            else:
                shutil.move(image_path, alt_folder)
    except:
        print('problem')
        pass
# [shutil.move('/Users/apple/Desktop/' + img, target_folder) for img in os.listdir('/Users/apple/Desktop/') if 'screenshot' and date_taken in img]
# predict_image('/Users/apple/Desktop/Screen Shot 2019-09-04 at 10.20.01 AM.png')