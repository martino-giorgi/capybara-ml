import numpy as np
from keras.models import load_model
import tensorflow as tf
from PIL import Image
import cv2
import random

model = load_model('deliverable/model.h5')

class_names = ['class_both_feet_combined_image', 'class_both_hands_combined_image', 'class_cervical_spine_lateral_image', 'class_left_foot_image', 'class_left_hand_image', 'class_lumbar_spine_ap_image', 'class_lumbar_spine_lateral_image', 'class_pelvis_ap_image', 'class_pelvis_lumbar_spine_combined_image', 'class_right_foot_image', 'class_right_hand_image']

url = "https://previews.123rf.com/images/tonporkasa/tonporkasa1910/tonporkasa191000258/137054736-blue-tone-both-hand-xray-on-white-background.jpg"
img_width = 300
img_height = 300

file_path = f"{random.random()*10}.png"
image = tf.keras.preprocessing.image.load_img(tf.keras.utils.get_file(file_path, cache_dir="./cache/", origin=url))

image = np.asarray(image)
image = (np.maximum(image, 0) / image.max()) * 255.0
image = np.uint8(image)
image = cv2.resize(image, (img_width, img_height))

png_image = Image.fromarray(image)
png_image.save('resized_image.png')

image = image.reshape(1,img_width,img_height,3)

predictions = model.predict(image)

score = tf.nn.softmax(predictions[0])
print(score)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)