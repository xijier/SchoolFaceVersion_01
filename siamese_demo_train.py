from keras_face.library.siamese import SiameseFaceNet
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from cv2 import *
import threading
import time
from PyQt5.QtGui import  QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QGroupBox, QPushButton, QLabel, QHBoxLayout,  QVBoxLayout, QGridLayout, QFormLayout, QLineEdit, QTextEdit
from MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet
import imutils
from rectangleDrawThread import rectangleThread
from PIL import Image
from pypinyin import pinyin, lazy_pinyin
import pypinyin
import pickle
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

def main(fnet):
    # fnet = SiameseFaceNet()
    # fnet.vgg16_include_top = True
    #
    #model_dir_path = './models_2'
    #image_dir_path = "./data/images"
    #
    #database = dict()

    #database["aipengfei"] = [fnet.img_to_encoding(image_dir_path + "/aipengfei.png")]
    #database["anyaru"] = [fnet.img_to_encoding(image_dir_path + "/anyaru.png")]
    #database["baozhiqian"] = [fnet.img_to_encoding(image_dir_path + "/baozhiqian.png")]

    # output = open('myfile.pkl', 'wb')
    # pickle.dump(database, output)
    # output.close()

    # database["danielle"] = [fnet.img_to_encoding(image_dir_path + "/danielle.png")]
    # database["younes"] = [fnet.img_to_encoding(image_dir_path + "/younes.jpg")]
    # database["tian"] = [fnet.img_to_encoding(image_dir_path + "/tian.jpg")]
    # database["andrew"] = [fnet.img_to_encoding(image_dir_path + "/andrew.jpg")]
    # database["kian"] = [fnet.img_to_encoding(image_dir_path + "/kian.jpg")]
    # database["dan"] = [fnet.img_to_encoding(image_dir_path + "/dan.jpg")]
    # database["sebastiano"] = [fnet.img_to_encoding(image_dir_path + "/sebastiano.jpg")]
    # database["bertrand"] = [fnet.img_to_encoding(image_dir_path + "/bertrand.jpg")]
    # database["kevin"] = [fnet.img_to_encoding(image_dir_path + "/kevin.jpg")]
    # database["felix"] = [fnet.img_to_encoding(image_dir_path + "/felix.jpg")]
    # database["benoit"] = [fnet.img_to_encoding(image_dir_path + "/benoit.jpg")]
    # database["arnaud"] = [fnet.img_to_encoding(image_dir_path + "/arnaud.jpg")]

    database = readPkl()
    model_dir_path = './models_2'
    fnet.fit(database=database, model_dir_path=model_dir_path,epochs=1000, batch_size=128)

def saveTrainPkl(fnet):
    database = dict()
    rootdir = 'data/stu100'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for index in range(0, len(list)):
        path = rootdir + "/" + list[index]
        listname = os.listdir(path)
        for i in range(0, len(listname)):
            print(path + '/' + listname[i])
            database[list[index]] = [fnet.img_to_encoding(path + '/' + listname[i])]
    output = open('myfileTraining_2609X25.pkl', 'wb')
    pickle.dump(database, output)
    output.close()

def readPkl():
    pkl_file = open('myfileTraining_2609X25.pkl', 'rb')
    database = pickle.load(pkl_file)
    pkl_file.close()
    return database

def createAug():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    rootdir = 'data/stu'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    count = 0
    for index in range(0, len(list)):
        print (count)
        count = count +1
        path = rootdir + "/" + list[index]
        listname = os.listdir(path)
        for i in range(0, len(listname)):
            #print(path + '/' + listname[i])
            imgpath = path + '/' + listname[0]
            img = load_img(imgpath)  # 这是一个PIL图像
            x = img_to_array(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
            x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)
            #  下面是生产图片的代码
            #  生产的所有图片保存在 `preview/` 目录下
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir=path+ "/", save_prefix=listname[i].split('.')[0], save_format='png'):
                i += 1
                if i > 25:
                    break  # 否则生成器会退出循环
def createAug_1():
    rootdir = 'data/stu01'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    count = 0
    for index in range(0, len(list)):
        #print (count)
        count = count +1
        path = rootdir + "/" + list[index]
        listname = os.listdir(path)
        if len(listname) == 0:
            print (list[index])

        #os.makedirs( 'data/stu01/'+list[index])
        for i in range(0, len(listname)):
            #print(path + '/' + listname[i])
            imgpath = path + '/' + listname[0]
            img = cv2.imread(imgpath)  # 这是一个PIL图像
            #img =cv2.cvtColor( np.array(img), cv2.COLOR_BGR2RGB)
            #cv2.imwrite('data/stu01/' +list[index]+'/'+ listname[0], img)
            break
if __name__ == '__main__':
    #createAug_1()
    fnet = SiameseFaceNet()
    fnet.vgg16_include_top = True
    saveTrainPkl(fnet)
    main(fnet)
    createAug()