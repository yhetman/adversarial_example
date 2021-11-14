import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
import seaborn as sns
from sklearn.metrics import accuracy_score


from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Reshape, BatchNormalization, Dropout, Conv2D, MaxPooling2D

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping


def load_images(images_folder, img_size = (64,64), scale=True, pred_set=False):
    image_path = []
    for dirname, _, filenames in os.walk(images_folder):
        for filename in filenames:
            image_path.append(os.path.join(dirname, filename))

    print("There are {} images in {}".format(len(image_path), images_folder))
    images = []
    labels = []

    for path in tqdm.tqdm(image_path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)    
        img = cv2.resize(img, img_size)
        img = np.array(img)
        images.append(img)
        if not pred_set:
            labels.append(categories[path.split('/')[-2]])
    images = np.array(images)  
    images = images.astype(np.int64)
    
    if scale:
        images = images / 255
        
    return image_path, images, np.asarray(labels)


def build_cnn_model():
    cnn_model=tf.keras.Sequential([
        Conv2D(filters=16,kernel_size=(3,3),activation='relu', \
             input_shape=intel_train_images.shape[1:]),
        MaxPooling2D((2,2), padding='same'),
        # BatchNormalization(),
        # Dropout(0.4),
        # Conv2D(16, 3, padding='same', activation='relu'),
        # MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(6)
      #Dense(6, activation='relu'),
      # Dense(6)
    ])

    return cnn_model

def predict_class(img):
    img = img.reshape(IMG_SIZE)
    predictions = model.predict(img)
    true_prediction = [tf.argmax(pred) for pred in predictions]
    true_prediction = np.array(true_prediction)
    return list(categories.keys())[list(categories.values()).index(true_prediction)]
    
    

categories = {
    'buildings': 0,
    'forest': 1,
    'glacier': 2,
    'mountain': 3,
    'sea': 4,
    'street': 5
}

BATCH_SIZE = 16
EPOCHS = 30
IMG_SIZE = (-1 ,64, 64, 3)
img_size = (64, 64)
images_train_folder = os.path.join('../', 'intel-image-classification', 'seg_train')
images_test_folder = os.path.join('../', 'intel-image-classification', 'seg_test')
# images_pred_folder = os.path.join('../', 'intel-image-classification', 'seg_pred', 'seg_pred')


image_path, intel_train_images, y_train = load_images(images_train_folder, img_size=img_size)
intel_train_images = np.array(intel_train_images).reshape(IMG_SIZE)

_, X_test, y_test = load_images(images_test_folder, img_size=img_size)
X_test = np.array(X_test).reshape(IMG_SIZE)

# _, X_pred, y_pred = load_images(images_pred_folder, img_size=img_size, scale=True, pred_set=True)
# X_pred = np.array(X_pred).reshape(IMG_SIZE)

# try:
model = tf.keras.models.load_model("./intel_image_classifier_relu3.h5")
# except:
#     model = build_cnn_model()
    
print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


checkpoint_filepath = './checkpoints/checkpoint_colors.hdf5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_freq=56300)


early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=10, 
    min_delta=0.001, 
    mode='max',
    restore_best_weights=True
)

history = model.fit(intel_train_images, y_train, 
                    batch_size = BATCH_SIZE, 
                    epochs = EPOCHS, 
                    verbose = 1, 
                    validation_data = (X_test, y_test),
                    callbacks=[model_checkpoint_callback, early_stopping])

model.save("intel_image_classifier_relu23.h5")

# model_preds = model.predict(X_test)
# model_preds = np.argmax(model_preds,axis=1)
# accuracy_score(y_test, model_preds)
