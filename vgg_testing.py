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



img = image_utils.load_img('test/1.png', target_size=(32, 32))
img = img.resize((32, 32), Image.ANTIALIAS)
# img = img.convert('L')  #转换为黑白
img = image_utils.img_to_array(img)

img = abs(255 - img)
img = np.expand_dims(img, axis=0)
img = img.astype('float32')
img /= 255
prediction = model.predict(img)
prediction = np.argmax(prediction, axis=1)
print(str(prediction[0]))

