import tensorflow as tf
from datetime import datetime
import pickle
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from spatial_transformer import spatial_transform_input
from utils import draw_samples


def create_localization_head(inputs):
    x = Conv2D(6, (5, 5), padding="valid", activation="relu")(inputs)
    x = MaxPooling2D((2, 2), strides=2)(x)
    x = Conv2D(16, (3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(
        6,
        activation="linear",
        kernel_initializer="zeros",
        bias_initializer=lambda shape, dtype: tf.constant([1, 0, 0, 0, 1, 0], dtype=dtype),
    )(
        x
    )  # 6 elements to describe the transformation

    return tf.keras.Model(inputs, x)


def create_mnist_baseline_model(input_shape):
    model_baseline = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    return model_baseline


def create_spatial_transform_network(input_shape):
    inputs = Input(shape=input_shape)
    localization_head = create_localization_head(inputs)
    mnist_baseline = create_mnist_baseline_model(input_shape)
    x = spatial_transform_input(inputs, localization_head.output)
    x = mnist_baseline(x)
    stn = tf.keras.Model(inputs, x)

    return stn, localization_head, mnist_baseline


def train(stn, x, y):
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    stn.fit(x, y, epochs=5, callbacks=[tensorboard_callback])


def get_mnist_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    input_shape = (28, 28, 1)

    distorted_mnist = pickle.load(open("distorted_mnist.p", "rb"))
    x_train_aug = distorted_mnist["x_train_distorted"]
    x_test_aug = distorted_mnist["x_test_distorted"]

    x_train, y_train, x_test, y_test = get_mnist_dataset()
    x = tf.concat([x_train_aug, x_train], 0)
    y = tf.concat([y_train, y_train], 0)

    stn, loc_head, mnist_model = create_spatial_transform_network(input_shape)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    stn.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    TRAIN = True
    if TRAIN:
        train(stn, x, y)
        stn.save_weights("./stn_weights")
        loc_head.save_weights("./loc_head")
        mnist_model.save_weights("./mnist_model")
    else:
        stn.load_weights("./stn_weights")
        loc_head.load_weights("./loc_head")
        mnist_model.load_weights("./mnist_model")

    print("Evaluating [distorted_mnist]:")
    stn.evaluate(x_test_aug, y_test)
    print("Evaluating [original_mnist]:")
    stn.evaluate(x_test, y_test)

    test_data = x_test_aug[:30]
    label = y_test[:30]
    draw_samples(test_data, "distorted_mnist", save_fig=True)

    predicted_transforms = loc_head.predict(test_data[:40])
    transformed_inputs = spatial_transform_input(test_data, predicted_transforms)
    draw_samples(transformed_inputs, "stn_corrected_mnist", save_fig=True)
