import os
import random
import shutil

source_folder = "/Users/apple/PycharmProjects/AutomaticScreenshotClassification/Other Screenshots"
target_folder_a = "/Users/apple/PycharmProjects/AutomaticScreenshotClassification/image_files/training_set/non_comments"
target_folder_b = "/Users/apple/PycharmProjects/AutomaticScreenshotClassification/image_files/test_set/non_comments"
split = 0.75  # the first split * 100% of images will get moved to target_folder_a, else go to b

image_list = os.listdir(source_folder)
random.shuffle(image_list)
split_index = int(split * len(image_list))

# move the first half to training set and the rest to test_set
for i in range(split_index):
    shutil.move(source_folder + '/' + image_list[i], target_folder_a)
for i in range(split_index, len(image_list)):
    shutil.move(source_folder + '/' + image_list[i], target_folder_b)


