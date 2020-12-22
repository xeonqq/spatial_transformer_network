import tensorflow as tf
import pickle
import imgaug.augmenters as iaa
import imgaug as ia


def generate_affine_distorted_mnist():
    ia.seed(42)
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    sometimes = lambda aug: iaa.Sometimes(1, aug)
    # Pipeline:
    # (1) Crop images from each side by 1-16px, do not resize the results
    #     images back to the input size. Keep them at the cropped size.
    # (2) Horizontally flip 50% of the images.
    # (3) Blur images using a gaussian kernel with sigma between 0.0 and 3.0.
    seq = iaa.Sequential(
        [
            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            iaa.OneOf(
                [
                    sometimes(
                        iaa.Affine(
                            scale={"x": (0.5, 1.3), "y": (0.5, 1.3)},
                            translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                            rotate=(-75, 75),
                            shear=(-30, 30),
                            order=[0, 1],
                            cval=(0),
                        )
                    ),
                    sometimes(
                        iaa.Affine(
                            scale={"x": (0.5, 1.3), "y": (0.5, 1.3)},
                            order=[0, 1],
                            cval=(0),
                        )
                    ),
                    sometimes(
                        iaa.Affine(
                            translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                            order=[0, 1],
                            cval=(0),
                        )
                    ),
                    sometimes(
                        iaa.Affine(
                            rotate=(-75, 75),
                            # shear=(-30, 30),
                            order=[0, 1],
                            cval=(0),
                        )
                    ),
                    sometimes(
                        iaa.Affine(
                            shear=(-30, 30),
                            order=[0, 1],
                            cval=(0),
                        )
                    ),
                    # In some images move pixels locally around (with random
                    # strengths).
                    #         sometimes(
                    #                     iaa.ElasticTransformation(alpha=(0., 1), sigma=0.25)
                    #         ),
                ]
            )
            # random_order=True,
        ]
    )
    x_train_aug = seq(images=x_train)  # done by the library
    x_test_aug = seq(images=x_test)  # done by the library

    distorted_mnist = {
        "x_train_distorted": x_train_aug,
        "x_test_distorted": x_test_aug,
        "y_train": y_train,
        "y_test": y_test,
    }
    pickle.dump(distorted_mnist, open("distorted_mnist.p", "wb"))


if __name__ == "__main__":
    generate_affine_distorted_mnist()
