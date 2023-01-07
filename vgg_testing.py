import tensorflow as tf
from keras.models import load_model


model = load_model('test/model.h5')
model.summary()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]])/255
x_test = tf.stack((x_test, x_test, x_test), axis=-1)

predict = model.predict(x_test)
print(predict)

score = model.evaluate(x_test, y_test)
print("Accuracy after loading Model:", score[1]*100)
