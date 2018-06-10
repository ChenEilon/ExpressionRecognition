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


def data_preprocess(img_dir, csv_dir, max_samples=2000, file_prefix="affectnet_landmarks"):
    print("Start preprocess")
    filenames = [entry.name for entry in os.scandir(csv_dir) if
                 entry.name.lower().endswith(".csv") and entry.name.startswith(file_prefix)]
    x = []
    y = []
    for f in filenames:
        print("Reading file {0}".format(f))
        data_df = pd.read_csv(os.path.join(csv_dir, f))
        for i in range(min(data_df.shape[0], max_samples)):
            f = os.path.join(img_dir, str(data_df.loc[i, "subDirectory"]), str(data_df.loc[i, "filePath"]))
            if not os.path.isfile(f):
                continue
            img = imread(f, as_grey=True)
            img = resize(img, (32, 32))
            x.append(img)
            y.append(data_df.loc[i, "expression"])
    # x = np.asarray(x, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=-1)
    y = np.asarray(y, dtype=np.int32)
    print("Saving data")
    np.save(os.path.join("x.npy"), x)
    np.save(os.path.join("y.npy"), y)
    print("Done preprocess")


def filter_classes(x, y, classes):
    indices = np.where(sum(y == c for c in classes))
    return x[indices], y[indices]


def generate_model_1(input_shape, num_classes):
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
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def generate_model_2(input_shape, num_classes):
    num_filters = 32
    num_pool = 2
    num_conv = 3
    model = Sequential()
    model.add(Conv2D(num_filters, (num_conv, num_conv), activation="relu", input_shape=input_shape))
    model.add(Conv2D(num_filters, (num_conv, num_conv), activation="relu"))
    model.add(Conv2D(num_filters, (num_conv, num_conv), activation="relu"))
    model.add(Conv2D(num_filters, (num_conv, num_conv), activation="relu"))
    model.add(MaxPooling2D(pool_size=(num_pool, num_pool)))
    model.add(Dropout(0.25))
    # model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    # model.add(Dropout(0.8))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def train_model(x, y, model, num_classes, label_dict={}, epochs=20, batch_size=128, validation_split=0.10):
    if len(label_dict) > 0:
        y = np.asarray([label_dict[a] for a in y], dtype=np.int32)
    y = np_utils.to_categorical(y, num_classes).astype(np.float32)
    indices = np.arange(len(x))
    np.random.RandomState(RANDOM_SEED).shuffle(indices)
    x = x[indices]
    y = y[indices]
    class_totals = y.sum(axis=0)
    class_weight = class_totals.max() / class_totals
    history = model.fit(x,
                        y,
                        batch_size=batch_size,
                        class_weight=class_weight,
                        epochs=epochs,
                        verbose=2,
                        validation_split=validation_split)
    model.save("model.h5")
    model.save_weights("weights.h5")
    return history


def train_binary_models(x, y, generate_model, classes):
    metrics = []
    for c in classes:
        model = generate_model(x.shape[1:], 2)
        label_dict = {i: 0 for i in classes}
        label_dict[c] = 1
        history = train_model(x, y, model, 2, label_dict)
        metrics.append([history.history["acc"], history.history["val_acc"]])
        model.save("model_bin_{0}.h5".format(c))
        model.save_weights("weights_bin_{0}.h5".format(c))
    print("Binary classifiers:")
    for i in range(len(classes)):
        print("{0}: acc: {1} - val_acc: {2}".format(classes[i], metrics[i][0][-1], metrics[i][1][-1]))
    return metrics


# data_preprocess(r"E:\final_project_ee\Manually Annotated\Manually_Annotated_Images",
#                 r"C:\Users\Santos\Documents\GitHub\ExpressionRecognition\Affectnet",
#                 5000)

x = np.load("x_32.npy")
y = np.load("y_32.npy")
classes = [0, 1, 2, 3, 6]
generate_model = generate_model_2

num_classes = len(classes)
label_dict = {classes[i]: i for i in range(num_classes)}

x, y = filter_classes(x, y, classes)
model = generate_model(x.shape[1:], num_classes)
train_model(x, y, model, num_classes, label_dict, validation_split=0.04)

# train_binary_models(x, y, generate_model, classes)
