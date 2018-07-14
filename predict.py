from keras_face.library.siamese import SiameseFaceNet
import pickle

class PredictImg():
    def __init__(self):
        self.data = ""
    def fileName(self):
        fnet = SiameseFaceNet()
        model_dir_path = './models_2'
        fnet.load_model(model_dir_path)
        database = dict()
        rootdir = 'data/stu100'
        list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        for index in range(0, len(list)):
            path = rootdir+"/"+list[index]
            listname = os.listdir(path)
            for i in range(0, len(listname)):
                print(path+'/'+listname[i])
                database[list[index]] = [fnet.img_to_encoding(path+'/'+listname[i])]
            #print (path+'/'+listname[i])
        output = open('myfile.pkl', 'wb')
        pickle.dump(database, output)
        output.close()

        return database
    def readPkl(self):
        pkl_file = open('myfileTraining_100X25.pkl', 'rb')
        database = pickle.load(pkl_file)
        pkl_file.close()
        return database
    def predict(self,database):
        fnet = SiameseFaceNet()
        model_dir_path = './models_2'
        fnet.load_model(model_dir_path)
        print("-------------------------")
        # fnet.verify(image_dir_path + "/001.png", "aipengfei", database)
        # fnet.verify(image_dir_path + "/002.png", "aipengfei", database)
        # fnet.who_is_it(image_dir_path + "/001.png", database)
        #fnet.who_is_it(image_dir_path + "/002.png", database)
        fnet.who_is_it("data/test/chencongcong.png", database)
        fnet.who_is_it("data/test/chendandan.png", database)
        # fnet.who_is_it("data/test/chendaocheng.png", database)
        # fnet.who_is_it("data/test/chenfuyuan.png", database)
        # fnet.who_is_it("data/test/chenge.png", database)
        # fnet.who_is_it("data/test/chenguang.png", database)
        # fnet.who_is_it("data/test/chenguangwei.png", database)
        # fnet.who_is_it("data/test/chenguoyan.png", database)
        # fnet.who_is_it("data/test/chenhaiyan.png", database)
        # fnet.who_is_it("data/test/chenhao.png", database)