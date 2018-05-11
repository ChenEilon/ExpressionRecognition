import os
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.io import imread
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils


RANDOM_SEED = 77


def data_preprocess(img_dir, csv_dir, file_prefix="affectnet_landmarks"):
    print("Start preprocess")
    filenames = [entry.name for entry in os.scandir(csv_dir) if
                 entry.name.lower().endswith(".csv") and entry.name.startswith(file_prefix)]
    x = []
    y = []
    for f in filenames:
        print("Reading file {0}".format(f))
        data_df = pd.read_csv(os.path.join(csv_dir, f))
        for i in range(data_df.shape[0]):
            img = imread(os.path.join(img_dir, data_df["subDirectory"], data_df["filePath"]), as_grey=True)
            img = resize(img, (32, 32))
            x.append(img)
            y.append(data_df["expression"])
    x = np.asarray(x, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=-1)
    y = np.asarray(y, dtype=np.int32)
    print("Saving data")
    np.save(os.path.join("x.npy"), x)
    np.save(os.path.join("y.npy"), y)
    print("Done preprocess")


def generate_model(input_shape, num_classes):
    num_filters = 32
    num_pool = 2
    num_conv = 3
    model = Sequential()
    model.add(Conv2D(num_filters, (num_conv, num_conv), activation="relu", input_shape=input_shape))
    model.add(Conv2D(num_filters, (num_conv, num_conv), activation="relu"))
    model.add(MaxPooling2D(pool_size=(num_pool, num_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def train_model(x, y, model, num_classes, epochs=20, validation_split=0.10):
    y = np_utils.to_categorical(y, num_classes).astype(np.float32)
    indices = np.arange(len(x))
    np.random.RandomState(RANDOM_SEED).shuffle(indices)
    x = x[indices]
    y = y[indices]
    class_totals = y.sum(axis=0)
    class_weight = class_totals.max() / class_totals
    model.fit(x,
              y,
              batch_size=128,
              class_weight=class_weight,
              epochs=epochs,
              verbose=2,
              validation_split=validation_split)
    model.save("model.h5")
    model.save_weights("weights.h5")


data_preprocess(r"E:\final_project_ee\Manually Annotated\Manually_Annotated_Images",
                r"C:\Users\Santos\Documents\GitHub\ExpressionRecognition\Affectnet")

x = np.load("x.npy")
y = np.load("y.npy")
num_classes = 8
model = generate_model(x.shape[1:], num_classes)

train_model(x, y, model, num_classes)
