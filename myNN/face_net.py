import pandas as pd
# from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import *
# from keras.optimizers import SGD
# import numpy as np
# import os
import sys
sys.path.insert(0, "..")
import imagePreProcessing


def data_preprocess(data_df):
    # Data needs to be scaled to a small range like 0 to 1 for the neural
    # network to work well.
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale both the training inputs and outputs
    scaled = scaler.fit_transform(data_df)

    # Create new pandas DataFrame objects from the scaled data
    scaled_df = pd.DataFrame(scaled, columns=data_df.columns.values)

    # Save scaled data dataframes to new CSV files
    scaled_df.to_csv("scaled_data.csv", index=False)

    # Save scaling data
    scaling_data = pd.DataFrame([scaler.scale_, scaler.min_], columns=data_df.columns.values)
    scaling_data.to_csv("scaling_data.csv", index=False)

    return scaled_df, scaler


def scale_data(data_df, scaling_data_df):
    return (data_df * scaling_data_df.iloc[0, :] + scaling_data_df.iloc[1, :]).as_matrix()


def createCNN2(num_classes, num_img, size_img):
    """not good"""
    print('Build model...')
    max_features = size_img
    maxlen = num_img
    model = Sequential()
    # embedded_layer =Embedding(200, embedding_dim, input_length=180)(inputs)
    model.add(Conv1D(kernel_size=3, filters=250, strides=1, padding='valid', activation='relu',
                     input_shape=(max_features, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(kernel_size=3, filters=250, strides=1, padding='valid', activation='relu',
                     input_shape=(max_features, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(kernel_size=3, filters=250, strides=1, padding='valid', activation='relu',
                     input_shape=(max_features, 1)))
    model.add(Conv1D(kernel_size=3, filters=250, strides=1, padding='valid', activation='relu',
                     input_shape=(max_features, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(256, return_sequences=False))
    model.add(RepeatVector(180, ))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dense(68, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(180, activation='softmax'))
    model.add(Dense(num_classes, activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    # model.summary()


def createCNN3(num_classes, num_img, size_img):
    print('Build model...')
    max_features = size_img
    maxlen = num_img
    model = Sequential()
    # embedded_layer =Embedding(200, embedding_dim, input_length=180)(inputs)
    model.add(Conv1D(kernel_size=3, filters=250, strides=1, padding='valid', activation='relu',
                     input_shape=(max_features, 1)))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(LSTM(256, return_sequences=True))
    model.add(Flatten())
    # model.add(Dense(180,activation='softmax'))
    model.add(Dense(num_classes, activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def createCNNModel(num_classes, num_img, size_img):
    print('Build model...')
    max_features = size_img
    maxlen = num_img
    # batch_size = 32
    nb_filter = 250
    filter_length = 3
    hidden_dims = 250
    # nb_epoch = 2
    model = Sequential()
    model.add(Conv1D(kernel_size=5, filters=250, strides=1, padding='valid', activation='relu',
                     input_shape=(max_features, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(80, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def createCNN4(num_classes, num_img, size_img):
    print('Build model...')
    max_features = size_img
    maxlen = num_img
    nb_filter = 250
    filter_length = 3
    hidden_dims = 250
    model = Sequential()
    model.add(Conv1D(kernel_size=5, filters=250, strides=1, padding='valid', activation='relu',
                     input_shape=(max_features, 1)))
    # model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid', activation='relu', subsample_length=1, input_shape=(maxlen, max_features)))
    # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(kernel_size=3, filters=250, strides=1, padding='valid', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(kernel_size = 3, filters = 250, strides=1, padding='valid', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # We flatten the output of the conv layer, so that we can add a vanilla dense layer:
    model.add(Flatten())
    # We add a vanilla hidden layer:
    model.add(Dense(80, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Activation('relu'))
    model.add(Dense(8, activation='relu'))
    # model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def getModel1(img_size):
    model = Sequential()
    model.add(Dense(2000, input_dim=img_size, activation='relu', name='layer_1'))
    # model.add(Conv1D(100, 6, padding='valid', activation='relu',strides=1))
    # model.add(GlobalMaxPooling1D()) # we use max pooling:
    model.add(Dense(100, activation='relu', name='layer_3'))
    model.add(Dense(40, activation='relu', name='layer_4'))
    model.add(Dense(7, activation='softmax', name='output_layer'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def getModel2(img_size):
    model = Sequential()
    model.add(Dense(2000, input_dim=img_size, activation='relu', name='layer_1'))
    # model.add(Conv1D(100, 6, padding='valid', activation='relu',strides=1))
    # model.add(GlobalMaxPooling1D()) # we use max pooling:
    model.add(Dense(40, activation='relu', name='layer_3'))
    model.add(Dense(80, activation='relu', name='layer_4'))
    model.add(Dense(40, activation='relu', name='layer_5'))
    model.add(Dense(7, activation='softmax', name='output_layer'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def getModel3(img_size):
    model = Sequential()
    model.add(Dense(2000, input_dim=img_size, activation='relu', name='layer_1'))
    # model.add(Conv1D(100, 6, padding='valid', activation='relu',strides=1))
    # model.add(GlobalMaxPooling1D()) # we use max pooling:
    model.add(Dense(80, activation='relu', name='layer_3'))
    model.add(Dense(80, activation='relu', name='layer_4'))
    model.add(Dense(40, activation='relu', name='layer_5'))
    model.add(Dense(7, activation='softmax', name='output_layer'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def getDenseModel(img_size):
    model = Sequential()
    model.add(Dense(4000, input_dim=img_size, activation='relu', name='layer_1'))
    model.add(Dense(2000, activation='relu', name='layer_2'))
    model.add(Dense(1000, activation='relu', name='layer_3'))
    model.add(Dense(500, activation='relu', name='layer_4'))
    # model.add(Conv1D(100, 6, padding='valid', activation='relu',strides=1))
    # model.add(GlobalMaxPooling1D()) # we use max pooling:
    model.add(Dense(100, activation='relu', name='layer_5'))
    # model.add(Dense(40, activation='relu', name='layer_4'))
    model.add(Dense(7, activation='linear', name='output_layer'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])
    return model


def dimension_reduction_pca(df, components=100):
    """
    input - dataframe of features & wanted dimension of features
    output - trained PCA
    uses PCA from skylearn
    """
    # Standardize the Data
    features = list(df.columns.values)
    # Separating out the features
    x = df.loc[:, features].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    # dim reduction
    pca = PCA(components)
    pca.fit_transform(x)
    return pca


emotions = [1, 2]
csvPaths = [r"..\Affectnet\features_affectnet_landmarks_{0}.csv".format(i) for i in emotions]
train_df, test_df = imagePreProcessing.prepare_balanced_data(csvPaths, 2000)

print("Start scaling...")
scaled_train_data, scaler = data_preprocess(train_df)
scaled_test_data = scaler.transform(test_df)
print("End scaling...")

# process to workable dfs
X_train = scaled_train_data.iloc[:, :-7].as_matrix()  # data
Y_oh_train = scaled_train_data.iloc[:, -7:].as_matrix()  # labels
X_test = scaled_test_data[:, :-7]  # data
Y_oh_test = scaled_test_data[:, -7:]  # labels

# generate single expression label
# Y_train = pd.DataFrame()
# Y_test = pd.DataFrame()
# Y_train['exp'] = sum(i*Y_oh_train.iloc[:, i] for i in range(8))
# Y_test['exp'] = sum(i*Y_oh_test.iloc[:, i] for i in range(8))
# Y_train = np.array(Y_train['exp'])
# Y_test = np.array(Y_test['exp'])

print("Data is prepared!")
# img_size = 6552
# pca dim reduction for conv:
# pca = dimension_reduction_pca(training_data_df.iloc[:, :-8], img_size)
# X = pca.transform(training_data_df.iloc[:, :-8])
# X_test = pca.transform(test_data_df.iloc[:, :-8])
# print("(not) PCA done. call model:")

# X = np.expand_dims(X, axis=2) # reshape (569, 30) to (569, 30, 1) for CNN
# X_test = np.expand_dims(X_test, axis=2) # reshape (569, 30) to (569, 30, 1) for CNN

# Define the model
img_size = 6552
model = getModel2(img_size)

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='logs',
    write_graph=True,
    histogram_freq=0
)

# Train the model
print("Train model:")
model.fit(
    X_train,
    Y_oh_train,
    epochs=20,
    shuffle=True,
    verbose=2,
    callbacks=[logger]
)

test_error_rate, test_acc = model.evaluate(X_test, Y_oh_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
print("The accuracy for the test data set is: {}".format(test_acc))
