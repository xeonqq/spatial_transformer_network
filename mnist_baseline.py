import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# with convolution layer it can reach 99% accuracy, but for simplicity, we will use the following FC version.
# model = tf.keras.models.Sequential([
#  tf.keras.layers.Conv2D(32, (3,3), input_shape=(H, W,1),padding='valid',activation="relu"),
#  tf.keras.layers.MaxPooling2D((2, 2)),
#  tf.keras.layers.Conv2D(16, (3,3),padding='valid',activation="relu"),
#  tf.keras.layers.MaxPooling2D((2, 2)),
#  tf.keras.layers.Flatten(),
#  tf.keras.layers.Dense(100, activation='relu'),
#
#  tf.keras.layers.Dropout(0.2),
#  tf.keras.layers.Dense(10),
# ])

# copy paste from https://www.tensorflow.org/tutorials/quickstart/beginner
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5)
model.save("./mnist_fc_base_line_model")

# accuracy ca. 0.9768
model.evaluate(x_test, y_test, verbose=2)
