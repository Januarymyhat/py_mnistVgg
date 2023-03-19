from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen,\
    QColor, QSize
from PyQt5.QtCore import Qt


class PaintBoard(QWidget):
    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.InitData() #先初始化数据，再初始化界面
        self.InitView()

    def InitData(self):

        self.size = QSize(800, 800)

        #新建QPixmap作为画板，尺寸为__size
        self.board = QPixmap(self.size)
        self.board.fill(Qt.white) #用白色填充画板

        self.isEmpty = True #默认为空画板
        self.eraserMode = False #默认为禁用橡皮擦模式

        self.lastPos = QPoint(0, 0)#上一次鼠标位置
        self.currentPos = QPoint(0, 0)#当前的鼠标位置

        self.painter = QPainter()#新建绘图工具

        self.thickness = 50       #默认画笔粗细为50px
        self.penColor = QColor("black")#设置默认画笔颜色为黑色
        self.colorList = QColor.colorNames() #获取颜色列表

    def InitView(self):
        #设置界面的尺寸为__size
        self.setFixedSize(self.size)

    def Clear(self):
        #清空画板
        self.board.fill(Qt.white)
        self.update()
        self.isEmpty = True

    def ChangePenColor(self, color="black"):
        #改变画笔颜色
        self.penColor = QColor(color)

    def ChangePenThickness(self, thickness=50):
        #改变画笔粗细
        self.thickness = thickness

    def IsEmpty(self):
        #返回画板是否为空
        return self.isEmpty

    def GetContentAsQImage(self):
        #获取画板内容（返回QImage）
        image = self.board.toImage()
        return image

    def paintEvent(self, paintEvent):
        #绘图事件
        #绘图时必须使用QPainter的实例，此处为__painter
        #绘图在begin()函数与end()函数间进行
        #begin(param)的参数要指定绘图设备，即把图画在哪里
        #drawPixmap用于绘制QPixmap类型的对象
        self.painter.begin(self)
        # 0,0为绘图的左上角起点的坐标，__board即要绘制的图
        self.painter.drawPixmap(0, 0, self.board)
        self.painter.end()

    def mousePressEvent(self, mouseEvent):
        #鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.currentPos = mouseEvent.pos()
        self.lastPos = self.currentPos


    def mouseMoveEvent(self, mouseEvent):
        #鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.currentPos =  mouseEvent.pos()
        self.painter.begin(self.board)

        if self.eraserMode == False:
            #非橡皮擦模式
            self.painter.setPen(QPen(self.penColor, self.thickness)) #设置画笔颜色，粗细
        else:
            #橡皮擦模式下画笔为纯白色，粗细为10
            self.painter.setPen(QPen(Qt.white, 80))

        #画线
        self.painter.drawLine(self.lastPos, self.currentPos)
        self.painter.end()
        self.lastPos = self.currentPos

        self.update() #更新显示

    def mouseReleaseEvent(self, mouseEvent):
        self.isEmpty = False #画板不再为空

