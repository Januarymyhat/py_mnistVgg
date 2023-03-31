import os
import shutil

from keras.models import load_model
import numpy as np
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPainter, QPixmap, QColor, QIcon
from PyQt5.QtGui import QFont
from PIL import ImageGrab, Image, ImageQt
from PyQt5.QtWidgets import QWidget, QCheckBox, QSpinBox, QComboBox, QHBoxLayout, QVBoxLayout, QFileDialog
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QLabel
from keras.utils import image_utils

import paint_board
import target_detection


class VGGUI(QWidget):

    def __init__(self):
        super(VGGUI, self).__init__()

        self.img = None
        self.setFixedSize(800, 950)  # 设置窗口宽高
        self.setWindowTitle("Digital Recognition")
        self.move(100, 100)  # 设置窗口出现时所处于屏幕的位置
        # self.setWindowFlags(Qt.FramelessWindowHint)  # 窗体无边框

        # self.file.setStyleSheet(
        #     "QPushButton{background-color:rgb(111,180,219)}"  # 按键背景色
        #     "QPushButton:hover{color:green}"  # 光标移动到上面后的前景色
        #     "QPushButton{border-radius:6px}"  # 圆角半径
        #     "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"  # 按下时的样式
        # )

        # 添加一系列控件
        self.paintBoard = paint_board.PaintBoard(self)
        # 上传文件显示的board
        self.label_showBoard = QLabel(self)

        # 获取QT中的颜色列表(字符串的List)
        self.colorList = QColor.colorNames()
        self.eraserMode = False  # 默认为禁用橡皮擦模式

        self.label_result_name = QLabel('Result: ', self)
        self.label_result_name.setGeometry(2, 810, 100, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(100, 810, 35, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:2px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)

        self.btn_recognize = QPushButton("Recognition", self)
        self.btn_recognize.setGeometry(160, 810, 160, 35)
        self.btn_recognize.clicked.connect(self.btn_recognize_on_clicked)

        self.btn_clear = QPushButton("Clear", self)
        self.btn_clear.setGeometry(350, 810, 80, 35)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)  # 绑定多个函数

        self.cbtn_eraser = QCheckBox("Eraser")
        self.cbtn_eraser.setParent(self)
        self.cbtn_eraser.setGeometry(460, 810, 100, 35)
        self.cbtn_eraser.clicked.connect(self.cbtn_eraser_on_clicked)

        self.btn_file = QPushButton("File", self)
        self.btn_file.setGeometry(580, 810, 100, 35)
        self.btn_file.clicked.connect(self.btn_file_on_click)

        self.label_penThickness = QLabel(self)
        self.label_penThickness.setText("Brush Size")
        self.label_penThickness.setGeometry(160, 860, 80, 35)
        self.spinBox_penThickness = QSpinBox(self)
        self.spinBox_penThickness.setGeometry(220, 860, 80, 35)
        self.spinBox_penThickness.setMaximum(100)
        self.spinBox_penThickness.setMinimum(40)
        self.spinBox_penThickness.setValue(50)  # 默认粗细为10
        self.spinBox_penThickness.setSingleStep(5)  # 最小变化值为2
        self.spinBox_penThickness.valueChanged.connect(
            self.on_PenThicknessChange)  # 关联spinBox值变化信号和函数on_PenThicknessChang

        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("Color")
        self.__label_penColor.setGeometry(320, 860, 80, 35)
        self.__label_penColor.setFixedHeight(20)
        self.__comboBox_penColor = QComboBox(self)
        self.__comboBox_penColor.setGeometry(380, 860, 80, 35)
        self.fillColorList(self.__comboBox_penColor)  # 用各种颜色填充下拉列表
        self.__comboBox_penColor.currentIndexChanged.connect(
            lambda: self.on_PenColorChange())  # 关联下拉列表的当前索引变更信号与函数on_PenColorChange

        self.__btn_close = QPushButton("Close", self)
        self.__btn_close.setGeometry(530, 860, 100, 35)
        self.__btn_close.clicked.connect(self.btn_close_on_clicked)
        # self.model = load_model('test/model.h5')

        self.img_number = 0
        self.img_processed_dir = 'test/detection/'
        self.__btn_back = QPushButton("Back", self)
        self.__btn_back.setGeometry(310, 900, 80, 35)
        self.__btn_back.clicked.connect(self.btn_back_on_click)
        self.__btn_back.hide()

        self.__btn_next = QPushButton("Next", self)
        self.__btn_next.setGeometry(420, 900, 80, 35)
        self.__btn_next.clicked.connect(self.btn_next_on_click)
        self.__btn_next.hide()

    # 关闭按钮的功能：关闭窗口
    def btn_close_on_clicked(self):
        self.close()

    def on_PenThicknessChange(self):
        thickness = self.spinBox_penThickness.value()
        self.paintBoard.ChangePenThickness(thickness)

    def fillColorList(self, comboBox):
        index_black = 0
        index = 0
        for color in self.colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.colorList[color_index]
        self.paintBoard.ChangePenColor(color_str)

    def cbtn_eraser_on_clicked(self):
        if self.cbtn_eraser.isChecked():
            self.paintBoard.eraserMode = True  # 进入橡皮擦模式
        else:
            self.paintBoard.eraserMode = False  # 退出橡皮擦模式

        # 识别按钮的功能：截屏手写数字并将截图转换成28*28像素的图片，之后调用识别函数并显示识别结果

    def btn_recognize_on_clicked(self):
        if not self.paintBoard.isEmpty:
            self.img = self.paintBoard.GetContentAsQImage()
            self.img.save(r'test/1.png')
            self.img = image_utils.load_img('test/1.png', target_size=(32, 32))

        predict = self.recognize_result(self.img)
        self.label_result.setText(str(predict[0]))  # 显示识别结果

    @staticmethod
    def recognize_result(img):
        model = load_model('test/model.h5')
        img = img.resize((32, 32), Image.ANTIALIAS)
        # img = img.convert('L')  #转换为黑白
        img = image_utils.img_to_array(img)

        img = abs(255 - img)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32')
        img /= 255
        prediction = model.predict(img)
        prediction = np.argmax(prediction, axis=1)
        return prediction

    # 清除上传的图片
    def btn_clear_on_clicked(self):
        # self.label_showBoard.setPixmap(QPixmap(""))
        if self.paintBoard.isEmpty:
            self.label_showBoard.setPixmap(QPixmap(""))
            self.label_showBoard.hide()
        else:
            self.paintBoard.Clear()

    def show_img(self):
        image = QPixmap(self.img_processed_dir + str(self.img_number) + '.png').scaled(800, 800)
        self.label_showBoard.setPixmap(image)
        self.img = ImageQt.fromqpixmap(image)

    def btn_file_on_click(self):
        # file_image = QFileDialog.getOpenFileName(self, "File", "C:\\", 'Image files (*.jpg *.png *.jpeg)')
        # self.paintBoard.__board = QPixmap(file_image)
        self.label_showBoard.show()
        self.label_showBoard.setFixedSize(800, 800)
        self.label_showBoard.move(0, 0)
        # self.label_showBoard.end()
        fiDir, fiType = QFileDialog.getOpenFileName(None, "File", "C:\\", 'Image files (*.jpg *.png *.jpeg)')
        if fiDir == '':
            pass
        else:
            # 清空文件夹
            shutil.rmtree(self.img_processed_dir)
            os.mkdir(self.img_processed_dir)
            # 开始目标检测
            target_detection.target_detect(fiDir)

            if not os.path.exists(self.img_processed_dir + '1.png'):
                pass
            else:
                self.__btn_next.show()
            self.show_img()

    def btn_back_on_click(self):
        self.__btn_next.show()
        self.img_number -= 1
        if self.img_number == 0:
            self.__btn_back.hide()
        self.show_img()

    def btn_next_on_click(self):
        self.__btn_back.show()
        self.img_number += 1
        images = len(os.listdir(self.img_processed_dir))
        if self.img_number == images - 1:
            self.__btn_next.hide()
        self.show_img()

