import tensorflow as tf
import keras
from keras import layers, losses
import matplotlib.pyplot as plt
from keras.applications.vgg16 import layers
import os
# Showing warning and error only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


def build_vgg16_model():
    input_shape = (32, 32, 3)
    model = keras.models.Sequential()

    # Conv1
    # `padding='same' indicates that padding is performed, the value of which is calculated internally by the algorithm
    # based on the size of the convolution kernel, with the aim of making the output size equal to the input;
    # The aim is to make the output size equal to the input size, but only if the stride size = 1. If the stride size
    # is not 1, then the output size must be different from the input size.
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv2
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv3
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv4
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv5
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Transition from a convolution layer to a fully connected layer
    model.add(layers.Flatten())
    # Fc6
    model.add(layers.Dense(4096, activation='relu'))
    # Fc7
    model.add(layers.Dense(4096, activation='relu'))
    # Fc8
    model.add(layers.Dense(10, activation='softmax'))
    return model


model = build_vgg16_model()
# model.summary()
# optimizer = optimizers.Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

# Set the loss, optimizer, and metrics for the built neural network model
model.compile(optimizer='adam',
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# Get the data.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

# Nomination
x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]])/255
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]])/255

x_train = tf.stack((x_train, x_train, x_train), axis=-1)
x_test = tf.stack((x_test, x_test, x_test), axis=-1)


#     # Data standardisation
# x_train = x_train / 255.0
# x_test = x_test / 255.0
#     # train set / data
# x_train = np.expand_dims(x_train, axis=-1)
# x_test = np.expand_dims(x_test, axis=-1)
#     # resize
# x_train = tf.image.resize(x_train, [48, 48])
# x_test = tf.image.resize(x_test, [48, 48])

# The function returns a History object, which records the change of the value of loss and other indicators with epoch.
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)  # batch size = 32 (Default)

# Draw accuracy and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, 'k', label='Training acc')    # k = black
plt.title('Training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure()
plt.plot(epochs, val_acc, 'k', label='Validation acc')
plt.title('Validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Draw loss and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'k', label='Training loss')
plt.title('Training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.figure()
plt.plot(epochs, val_loss, 'k', label='Validation loss')
plt.title('Validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# save the model
model.save(r'test/model.h5')

