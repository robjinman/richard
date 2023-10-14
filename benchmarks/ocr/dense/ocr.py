import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
num_training_samples = 1000
num_test_samples = 1000
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[:num_training_samples,:,:]
y_train = y_train[:num_training_samples]
x_test = x_test[:num_test_samples,:,:]
y_test = y_test[:num_test_samples]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

initializer = keras.initializers.RandomNormal(mean=0., stddev=0.1)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(300, activation="sigmoid", kernel_initializer=initializer,
            bias_initializer="zeros"),
        layers.Dense(80, activation="sigmoid", kernel_initializer=initializer,
            bias_initializer="zeros"),
        #layers.Dropout(0.5),
        layers.Dense(num_classes, activation="sigmoid", kernel_initializer=initializer,
            bias_initializer="zeros"),
    ]
)

model.summary()

batch_size = 1
epochs = 30

optimizer = keras.optimizers.SGD(learning_rate=0.7)
print("Learn rate = ", optimizer.learning_rate)

model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
