'''Train a simple convnet on the part olivetti faces dataset.
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py
Get to 95% test accuracy after 25 epochs (there is still a lot of margin for parameter tuning).
'''

from __future__ import print_function
import numpy

numpy.random.seed(1337)  # for reproducibility

from PIL import Image

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from cv2 import *

from sklearn.model_selection import train_test_split

# There are 40 different classes
nb_classes = 40
nb_epoch = 40
batch_size = 40

# input image dimensions
img_rows, img_cols = 57, 47
# number of convolutional filters to use
nb_filters1, nb_filters2 = 5, 10
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#99个学生
#每个学生27张图片
def load_data(dataset_path):
    rootdir = '../data/stu100'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    facesList = numpy.empty((2673, 33124))
    for index in range(0, len(list)):
        path = rootdir + "/" + list[index]
        listname = os.listdir(path)
        for i in range(0, len(listname)):
            print(path + '/' + listname[i])
            img = Image.open(path + '/' + listname[i])
            img_ndarray = numpy.asarray(img, dtype='float64') / 256
            #cv2.imshow("1",img_ndarray)
            #cv2.waitKey(0)

    img = Image.open(dataset_path)
    img_ndarray = numpy.asarray(img, dtype='float64') / 256
    # 400pictures,size:57*47=2679
    faces = numpy.empty((400, 2679))
    i = 0
    for row in range(20):
        for column in range(20):
            faces[row * 20 + column] = numpy.ndarray.flatten(img_ndarray[row * 57:(row + 1) * 57, column * 47:(column + 1) * 47])
            i = i+ 1

    label = numpy.empty(400)
    for i in range(40):
        label[i * 10:i * 10 + 10] = i
    label = label.astype(numpy.int)

    # train:320,valid:40,test:40
    train_data = numpy.empty((320, 2679))
    train_label = numpy.empty(320)
    valid_data = numpy.empty((40, 2679))
    valid_label = numpy.empty(40)
    test_data = numpy.empty((40, 2679))
    test_label = numpy.empty(40)

    for i in range(40):
        train_data[i * 8:i * 8 + 8] = faces[i * 10:i * 10 + 8]
        train_label[i * 8:i * 8 + 8] = label[i * 10:i * 10 + 8]
        valid_data[i] = faces[i * 10 + 8]
        valid_label[i] = label[i * 10 + 8]
        test_data[i] = faces[i * 10 + 9]
        test_label[i] = label[i * 10 + 9]
    rval = [(train_data, train_label), (valid_data, valid_label),
            (test_data, test_label)]
    return rval


def Net_model(lr=0.005, decay=1e-6, momentum=0.9):
    model = Sequential()
    model.add(Conv2D(nb_filters1, (nb_conv, nb_conv),padding='valid',input_shape=(img_rows, img_cols,1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Conv2D(nb_filters2, (nb_conv, nb_conv)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1000))  # Full connection
    model.add(Activation('tanh'))
    # model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


def train_model(model, X_train, Y_train, X_val, Y_val):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, verbose=1,batch_size=batch_size, nb_epoch=nb_epoch,validation_data=(X_val, Y_val))
    model.save_weights('model_weights.h5', overwrite=True)
    return model


def test_model(model, X, Y):
    model.load_weights('model_weights.h5')
    score = model.evaluate(X, Y, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return score


if __name__ == '__main__':

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data('olivettifaces.gif')

    X_train = X_train.reshape(X_train.shape[0],  img_rows, img_cols ,1)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols,1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validate samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Net_model()
    train_model(model,X_train,Y_train,X_val,Y_val)
    #score=test_model(model,X_test,Y_test)

    model.load_weights('model_weights.h5')
    classes = model.predict_classes(X_test, verbose=0)
    test_accuracy = numpy.mean(numpy.equal(y_test, classes))
    print("accuarcy:", test_accuracy)