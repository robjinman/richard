import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def load_data_set(csv_file, num_samples, W, H):
    samples_x = []
    samples_y = []

    with open(csv_file, "r") as f:
        i = 0
        for line in f:
            if i == num_samples:
                break

            label, *str_data = line.split(",")
            flat_array = [ float(x) for x in str_data ]

            assert(len(flat_array) % W == 0)
            num_rows = int(len(flat_array) / W)

            sample = []

            for r in range(0, num_rows):
                offset = r * W
                row = flat_array[offset : offset + W]
                sample.append(row)

            samples_x.append(sample)
            samples_y.append(int(label))

            i += 1

    return np.array(samples_x), np.array(samples_y)


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets

x_train, y_train = load_data_set("../../../data/ocr/train.csv", 1000, 28, 28)
x_test, y_test = load_data_set("../../../data/ocr/test.csv", 1000, 28, 28)

# Scale images to the [0, 1] range
x_train = x_train / 255
x_test = x_test / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

initializer1 = keras.initializers.RandomNormal(mean=0., stddev=0.1)
initializer2 = keras.initializers.RandomNormal(mean=0., stddev=0.1)
initializer3 = keras.initializers.RandomNormal(mean=0., stddev=0.1)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(300, activation="sigmoid", kernel_initializer=initializer1,
            bias_initializer="zeros"),
        layers.Dense(80, activation="sigmoid", kernel_initializer=initializer2,
            bias_initializer="zeros"),
        #layers.Dropout(0.5),
        layers.Dense(num_classes, activation="sigmoid", kernel_initializer=initializer3,
            bias_initializer="zeros"),
    ]
)

model.summary()

batch_size = 1
epochs = 30

optimizer = keras.optimizers.SGD(learning_rate=0.7)

model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
