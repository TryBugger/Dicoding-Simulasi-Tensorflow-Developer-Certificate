# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # NORMALIZE YOUR IMAGE HERE
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train = X_train / 255.
    X_test = X_test / 255.

    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Conv2D(128, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Input((28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # COMPILE MODEL HERE
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=['accuracy'])

    # TRAIN YOUR MODEL HERE
    class StopAccuracyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') >= 0.83) & (logs.get('val_accuracy') >= 0.83):
                self.model.stop_training = True

    reduceRLOnPlateau = ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.75,
                                          patience=5,
                                          min_lr=1e-5,
                                          verbose=1)

    model.fit(X_train,
              y_train,
              validation_data=(X_test, y_test),
              epochs=20,
              verbose=1,
              callbacks=[StopAccuracyCallback(),
                         reduceRLOnPlateau])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
