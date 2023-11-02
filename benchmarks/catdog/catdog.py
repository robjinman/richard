import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def load_data_set(class1_dir, class2_dir, n):
  samples = list()
  labels = list()
  
  class1_files = [ os.path.join(class1_dir, f) for f in os.listdir(class1_dir) ]
  class2_files = [ os.path.join(class2_dir, f) for f in os.listdir(class2_dir) ]
  
  i = 0
  for class1_file, class2_file in zip(class1_files, class2_files):
    samples.append(img_to_array(load_img(class1_file)))
    labels.append(0)
    samples.append(img_to_array(load_img(class2_file)))
    labels.append(1)

    i += 2

    if i >= n:
      break
      
  return np.asarray(samples), np.asarray(labels)


num_classes = 2
input_shape = (100, 100, 3)

num_training_samples = 1000
cats_training_dir = "../../data/catdog/train/cat"
dogs_training_dir = "../../data/catdog/train/dog"
x_train, y_train = load_data_set(cats_training_dir, dogs_training_dir, num_training_samples)
x_train = x_train / 255

num_test_samples = 100
cats_test_dir = "../../data/catdog/test/cat"
dogs_test_dir = "../../data/catdog/test/dog"
x_test, y_test = load_data_set(cats_test_dir, dogs_test_dir, num_test_samples)
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

initializer1 = keras.initializers.RandomNormal(mean=0., stddev=0.1)
initializer2 = keras.initializers.RandomNormal(mean=0., stddev=0.1)
initializer3 = keras.initializers.RandomNormal(mean=0., stddev=0.1)
initializer4 = keras.initializers.RandomNormal(mean=0., stddev=0.1)

model = keras.Sequential(
  [
    keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu",
      kernel_initializer=initializer1, bias_initializer="zeros"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu",
      kernel_initializer=initializer2, bias_initializer="zeros"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="sigmoid", kernel_initializer=initializer3,
      bias_initializer="zeros"),
    layers.Dense(num_classes, activation="sigmoid", kernel_initializer=initializer4,
      bias_initializer="zeros"),
  ]
)

model.summary()

batch_size = 1
epochs = 10

optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9) # TODO: Remove momentum

model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=False)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

