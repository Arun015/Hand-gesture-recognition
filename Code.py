import h5py
import numpy as np
from keras import layers
from keras.layers import Input, Add,Dropout, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.applications.imagenet_utils import preprocess_input
import scipy.misc
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

%matplotlib inline
from keras import applications
import keras.backend as K
from keras.optimizers import SGD, Adam

# Loading the data (signs)
def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
X_train, Y_train, X_test, Y_test, classes = load_dataset()
X_train.shape

X_train = X_train / 255
X_test = X_test / 255

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
Y_train.shape
Y_train = convert_to_one_hot(Y_train, 6).T
Y_test = convert_to_one_hot(Y_test, 6).T
img_height,img_width = 64,64 
num_classes = 6

base_model = applications.resnet50.ResNet50( include_top=False, input_shape= (img_height,img_width,3)) #weights= None,
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs =1, batch_size = 64)

pred = model.evaluate(X_train, Y_train)
print ("Loss = " + str(pred[0]))
print ("Train Accuracy = " + str(pred[1]))
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
