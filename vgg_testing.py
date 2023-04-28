import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from keras.utils import image_utils
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Load the trained model
model = load_model('test/model.h5')
# model.summary()
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
# Preprocess the test dataset
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]) / 255
x_test = tf.stack((x_test, x_test, x_test), axis=-1)
# print(x_train.shape, y_train.shape)

# Get the model predictions on the test dataset
predict = model.predict(x_test)
# print(predict)
# Accuracy and Loss
# loss, accuracy = model.evaluate(x_test, y_test)
# print("Accuracy after loading Model:", accuracy*100)
# print("Loss after loading Model:", loss*100)


# Get model predictions on test dataset
y_pred = np.argmax(predict, axis=-1)
# Calculate confusion matrix for all digit classes
confusion = confusion_matrix(y_test, y_pred)
# Calculate TP, TN, FP, and FN counts for each digit class
TP = np.zeros(10)
FN = np.zeros(10)
FP = np.zeros(10)
TN = np.zeros(10)
for i in range(10):
    TP[i] = confusion[i][i]
    FN[i] = np.sum(confusion[i]) - TP[i]
    FP[i] = np.sum(confusion[:, i]) - TP[i]
    TN[i] = np.sum(confusion) - TP[i] - FN[i] - FP[i]
    print("Digit {}: TP={}, TN={}, FP={}, FN={}".format(i, TP[i], TN[i], FP[i], FN[i]))

# Calculate accuracy, precision, recall, and F1 score for each digit class
accuracy = TP + TN / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)
for i in range(10):
    print("Digit {}: accuracy={}, precision={}, recall={}, F1 score={}"
          .format(i, accuracy[i], precision[i], recall[i], f1_score[i]))
    # Calculate the TPR and FPR for each digit class and plot the ROC curve for the entire dataset
    y_pred_proba = model.predict(x_test)[:, i]
    fpr, tpr, thresholds = roc_curve(y_test == i, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    # plot the ROC curve
    plt.plot(fpr, tpr, 'k', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of ' + 'Digit {}'.format(i))
    plt.legend(loc="lower right")
    plt.show()




# assuming y_test and y_pred are the true labels and predicted probabilities of the binary classification model
# # Get the probabilities for class 1 (digit '0')
# y_pred_proba = predict[:, 0]
# Compute the ROC curve and AUC score for class 1 (digit '0')
# fpr, tpr, thresholds = roc_curve(y_test == 0, predict)
# roc_auc = auc(fpr, tpr)


# Show dataset
# plt.imshow(x_test[0])
# plt.show()
# plt.imshow(x_test[1])
# plt.show()
# plt.imshow(x_test[2])
# plt.show()
# plt.imshow(x_test[3])
# plt.show()

# # detection test
# img = image_utils.load_img('test/1.png', target_size=(32, 32))
# img = img.resize((32, 32), Image.ANTIALIAS)
# # img = img.convert('L')  #转换为黑白
# img = image_utils.img_to_array(img)
#
# img = abs(255 - img)
# img = np.expand_dims(img, axis=0)
# img = img.astype('float32')
# img /= 255
# prediction = model.predict(img)
# prediction = np.argmax(prediction, axis=1)
# print(str(prediction[0]))

