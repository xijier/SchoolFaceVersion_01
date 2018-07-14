import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QMenuBar, QToolBar, QTextEdit, QAction, QApplication, qApp, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QApplication, QGroupBox, QPushButton, QLabel, QHBoxLayout,  QVBoxLayout, QGridLayout
from PyQt5.QtGui import  QPixmap
from queue import Queue

class ContentWidget(QDialog):
    def __init__(self, parent=None):
        super(ContentWidget, self).__init__(parent)
        self.setStyleSheet("background: black")


class IndexWidget(QDialog):
    def __init__(self, parent=None):
        super(IndexWidget, self).__init__(parent)
        self.setStyleSheet("background: red")


class TabWidget(QTabWidget):
    def __init__(self, parent=None):
        super(TabWidget, self).__init__(parent)
        self.resize(400, 300)
        self.mContent = ContentWidget()
        self.mIndex = IndexWidget()
        self.addTab(self.mContent, u"内容")
        self.addTab(self.mIndex, u"索引")


if __name__ == '__main__':
    import sys
    # app = QApplication(sys.argv)
    # t = TabWidget()
    # t.show()
    # app.exec_()
    q = Queue()  # 创建队列对象
    q.put(0)  # 在队列尾部插入元素
    q.put(1)
    q.put(2)
    print(q.get())  # 返回并删除队列头部元素
    q.put(3)
    print(q.queue)
