import datetime
import pandas as pd
import os
import shutil
from matplotlib import pyplot as plt
import pydicom
import numpy as np
from PIL import Image
from pathlib import Path
import sklearn
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import cv2

from keras.models import Sequential
from keras.layers import Dense, RandomBrightness, RandomZoom, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, RandomRotation, RandomTranslation, RandomContrast, BatchNormalization, GlobalAveragePooling2D

from utils.utils import make_confusion_matrix, plot_to_image


os.environ['CUDA_VISIBLE_DEVICES'] = "1"

batch_size = 1
epochs = 100
img_width = 300
img_height = 300


data_augmentation = tf.keras.Sequential([
    RandomRotation(0.1, fill_mode='nearest'),
    RandomTranslation(0.05,0.05, fill_mode='nearest'),
    RandomContrast(0.2),
    RandomZoom(0.1),
    RandomBrightness(0.2)
    
])


def load_model():
    nb_classes = 11

    model = Sequential()
    model.add(data_augmentation)
    model.add(Conv2D(32, (5, 5), padding="valid", strides=(2, 2), input_shape=(
        img_width, img_height, 3)))  # output=((227-5)/2 + 1 = 112
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
              )  # output=((112-2)/2 + 1 = 56

    model.add(Conv2D(32, (5, 5), padding="same"))  # output = 56
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding="same"))  # output = 56
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
              )  # output=((56-2)/2 + 1 = 28

    model.add(Conv2D(64, (3, 3), padding="same"))  # output = 28
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding="same"))  # output= 28
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
              )  # output=((28-2)/2 + 1 = 14

    model.add(Conv2D(96, (3, 3), padding="same"))  # output = 14
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # output = ((14-3)/1) +1 = 12
    model.add(Conv2D(96, (3, 3), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
              )  # output=((12-2)/2 + 1 = 6

    model.add(Conv2D(192, (3, 3), padding="same"))  # output =6
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # output = ((6-3)/1) + 1 = 4
    model.add(Conv2D(192, (3, 3), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
              )  # output=((4-2)/2 + 1 = 2

    model.add(Flatten())

    model.add(Dense(units=4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))  # for first level

    model.add(Dense(units=4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))  # for first level

    model.add(Dense(units=nb_classes))
    model.add(Activation('softmax'))

    return model


def load_model_tl():
    demo_resnet_model = Sequential()
    demo_resnet_model.add(data_augmentation)
    pretrained_model_for_demo = tf.keras.applications.ResNetRS50(include_top=False,
                                                                  input_shape=(
                                                                      img_width, img_height, 3),
                                                                  weights='imagenet',
                                                                  classes=11)
    pretrained_model_for_demo.trainable = False

    demo_resnet_model.add(pretrained_model_for_demo)
    demo_resnet_model.add(Flatten())

    demo_resnet_model.add(Dense(units=1024))
    demo_resnet_model.add(Activation('relu'))
    demo_resnet_model.add(Dropout(0.4))

    demo_resnet_model.add(Dense(units=512))
    demo_resnet_model.add(Activation('relu'))
    demo_resnet_model.add(Dropout(0.4))

    demo_resnet_model.add(Dense(units=11))
    demo_resnet_model.add(Activation('softmax'))

    return demo_resnet_model


def sort_images():
    df = pd.read_excel("/home/mialab23.team1/data/SCQM.xlsx",
                       usecols=["img_uid", "diagnostic_image_type"])

    labels = df['diagnostic_image_type'].unique()

    for label in labels:
        path = f"data/class_{label}"

        Path(path).mkdir(parents=True, exist_ok=True)
        [f.unlink() for f in Path(path).iterdir() if f.is_file()]

        images_uid = df.loc[df['diagnostic_image_type']
                            == label]['img_uid'].tolist()

        for image in images_uid:
            try:
                shutil.copy(
                    f"/home/mialab23.team1/data/SCQM/{image}.dcm", path)
                ds = pydicom.dcmread(f'{path}/{image}.dcm')
                new_image = ds.pixel_array.astype(float)
                new_image = (np.maximum(new_image, 0) /
                             new_image.max()) * 255.0
                new_image = np.uint8(new_image)
                new_image = cv2.resize(new_image, (img_width, img_height))

                final_image = Image.fromarray(new_image)

                final_image.save(f'{path}/{image}.jpeg')
                os.remove(f"{path}/{image}.dcm")
            except Exception as e:
                print(e)


# un-comment only first time
# sort_images()

train_ds, validation_ds = tf.keras.utils.image_dataset_from_directory(
    "data/",
    validation_split=0.20,
    label_mode='int',
    seed=12,
    shuffle=True,
    subset="both",
    image_size=(img_width, img_height))

validation_ds, test_ds = tf.keras.utils.split_dataset(
    validation_ds, left_size=0.5, shuffle=True)

# model = load_model()
model = load_model_tl()

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_accuracy',
                           patience=20, restore_best_weights=True)

logdir = "logs/image/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir, histogram_freq = 1)
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

test_images = np.concatenate([x for x, y in test_ds], axis=0)
test_labels = np.concatenate([y for x, y in test_ds], axis=0)
class_names = ['Both Feet Combined', 'Both Hands Combined', 'Cervical Spine Lateral', 'Left Foot', 'Left Hand', 'Lumbar Spine Ap', 'Lumbar Spine Lateral', 'Pelvis Ap', 'Pelvis Lumbar Spine Combined', 'Right Foot', 'Right Hand']

def log_confusion_matrix(epoch, logs):
    
    # Use the model to predict the values from the test_images.
    test_pred_raw = model.predict(test_images)
    
    test_pred = np.argmax(test_pred_raw, axis=1)
    
    # Calculate the confusion matrix using sklearn.metrics
    cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
    
    figure = make_confusion_matrix(cm, categories=class_names)
    cm_image = plot_to_image(figure)
    
    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

history = model.fit(train_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=validation_ds,
                    callbacks=[early_stop, tensorboard_callback, cm_callback])

score = model.evaluate(test_ds)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('deliverable/model.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy.png')
