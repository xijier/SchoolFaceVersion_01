from MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet
import cv2
class MTCNNLoader:
    def loadNet(self):
        global Pnet, Rnet, Onet
        Pnet = create_Kao_Pnet(r'12net.h5')
        Rnet = create_Kao_Rnet(r'24net.h5')
        Onet = create_Kao_Onet(r'48net.h5')  # will not work. caffe and TF incompatible
        img = cv2.imread('data/0001.png')
        scale_img = cv2.resize(img, (100, 100))
        input = scale_img.reshape(1, *scale_img.shape)
        Pnet.predict(input)
        img = cv2.imread('data/0001.png')
        scale_img = cv2.resize(img, (24, 24))
        input = scale_img.reshape(1, *scale_img.shape)
        Rnet.predict(input)
        img = cv2.imread('data/0001.png')
        scale_img = cv2.resize(img, (48, 48))
        input = scale_img.reshape(1, *scale_img.shape)
        Onet.predict(input)
        return Pnet, Rnet, Onet