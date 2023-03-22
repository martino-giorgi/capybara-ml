import pandas as pd
import os
import shutil
import pydicom
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, RandomRotation, RandomTranslation, RandomContrast, BatchNormalization


os.environ['CUDA_VISIBLE_DEVICES'] = "1"

batch_size = 16
epochs = 500
img_width=227
img_height=227

data_augmentation = tf.keras.Sequential([
    RandomRotation(0.4, fill_mode='nearest'),
    # RandomTranslation(0.2,0.2, fill_mode='nearest'),
    RandomContrast(0.4),
])

def load_model():
    nb_classes=11
    
    model = Sequential()
    model.add(data_augmentation)
    model.add(Conv2D(32,(5,5),padding="valid",strides=(2,2),input_shape=(img_width,img_height,3))) #output=((227-5)/2 + 1 = 112
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((112-2)/2 + 1 = 56
    

    model.add(Conv2D(32,(5,5),padding="same")) #output = 56
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3),padding="same"))  #output = 56
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((56-2)/2 + 1 = 28

    
    model.add(Conv2D(64,(3,3),padding="same"))  #output = 28
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3),padding="same")) #output= 28
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((28-2)/2 + 1 = 14
    
    
    
    model.add(Conv2D(96,(3,3),padding="same"))  #output = 14
    model.add(BatchNormalization())
    model.add(Activation('relu'))    
    model.add(Conv2D(96,(3,3),padding="valid"))  #output = ((14-3)/1) +1 = 12
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((12-2)/2 + 1 = 6
    
    

    model.add(Conv2D(192,(3,3),padding="same"))  #output =6
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3,3),padding="valid"))  #output = ((6-3)/1) + 1 = 4 
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((4-2)/2 + 1 = 2 
    
    model.add(Flatten())
    
    model.add(Dense(units=4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4)) # for first level
    # model.add(Dropout(0.4)) # for sec level
    
    model.add(Dense(units=4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4)) # for first level
    # model.add(Dropout(0.4)) # for sec level
    
    model.add(Dense(units=nb_classes))
    model.add(Activation('softmax'))
    
    return model


def sort_images():
    df = pd.read_excel("/home/mialab23.team1/data/SCQM.xlsx", usecols=["img_uid", "diagnostic_image_type"])

    labels = df['diagnostic_image_type'].unique()

    for label in labels:
        path = f"data/class_{label}"

        Path(path).mkdir(parents=True, exist_ok=True)
        [f.unlink() for f in Path(path).iterdir() if f.is_file()] 

        images_uid = df.loc[df['diagnostic_image_type'] == label]['img_uid'].tolist()

        for image in images_uid:
            try:
                shutil.copy(f"/home/mialab23.team1/data/SCQM/{image}.dcm", path)
                ds = pydicom.dcmread(f'{path}/{image}.dcm')
                new_image = ds.pixel_array.astype(float)
                new_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
                new_image = np.uint8(new_image)
                new_image = cv2.resize(new_image, (img_width, img_height))
                
                final_image = Image.fromarray(new_image)
                # final_image.thumbnail((img_width,img_height))
                # final_image = expand2square(final_image, 'black')
                
                final_image.save(f'{path}/{image}.png')
                os.remove(f"{path}/{image}.dcm")
            except Exception as e: print(e)

# sort_images()

train_ds, validation_ds = tf.keras.utils.image_dataset_from_directory(
  "data/",
  validation_split=0.20,
  label_mode='int',
  seed=12,
  shuffle=True,
  subset="both",
  image_size=(img_width, img_height),
  )


validation_ds, test_ds = tf.keras.utils.split_dataset(validation_ds, left_size=0.5, shuffle=True)

model = load_model()

model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

history = model.fit(train_ds,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=validation_ds,
                        callbacks=[early_stop])

score = model.evaluate(test_ds)

model.save('deliverable/model.h5')