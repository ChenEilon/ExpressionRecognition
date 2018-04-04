import pandas as pd
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import *

data_df = pd.read_csv("face_scaled.csv")
data_df = shuffle(data_df, random_state=50)

training_data_df = data_df[:500]
test_data_df = data_df[500:]

X = training_data_df.iloc[:, :-8].as_matrix()
Y = training_data_df.iloc[:, -8:].as_matrix()

X_test = test_data_df.iloc[:, :-8].as_matrix()
Y_test = test_data_df.iloc[:, -8:].as_matrix()

# Define the model
model = Sequential()
model.add(Dense(1000, input_dim=6930, activation='relu', name='layer_1'))
model.add(Dense(100, activation='relu', name='layer_2'))
model.add(Dense(40, activation='relu', name='layer_3'))
model.add(Dense(8, activation='linear', name='output_layer'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='logs',
    write_graph=True,
    histogram_freq=5
)

# Train the model
model.fit(
    X,
    Y,
    epochs=25,
    shuffle=True,
    verbose=2,
    callbacks=[logger]
)

test_error_rate, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
print("The accuracy for the test data set is: {}".format(test_acc))
