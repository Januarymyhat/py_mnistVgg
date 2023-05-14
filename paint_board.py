from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen,\
    QColor, QSize
from PyQt5.QtCore import Qt


class PaintBoard(QWidget):
    def __init__(self, Parent = None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        # Initialize the data before initializing the interface
        self.InitData()
        self.InitView()

    def InitData(self):

        self.size = QSize(800, 800)

        # A new QPixmap is created as a paint board
        self.board = QPixmap(self.size)
        # Fill the paint board with white
        self.board.fill(Qt.white)

        # Default the paint board is empty and disable eraser mode
        self.isEmpty = True
        self.eraserMode = False

        # Last mouse position
        self.lastPos = QPoint(0, 0)
        # The current mouse position
        self.currentPos = QPoint(0, 0)
        # Create new drawing tool
        self.painter = QPainter()

        # Default the brush thickness is 50 px
        self.thickness = 50
        # Default the brush color is black
        self.penColor = QColor("black")
        # Get a list of colors
        self.colorList = QColor.colorNames()

    def InitView(self):
        # Set the size of the interface
        self.setFixedSize(self.size)

    # Clear paint board
    def Clear(self):
        self.board.fill(Qt.white)
        self.update()
        self.isEmpty = True

    # Change brush color
    def ChangePenColor(self, color="black"):
        self.penColor = QColor(color)

    # Change brush thickness
    def ChangePenThickness(self, thickness=50):
        self.thickness = thickness

    # Return whether the paint board is empty
    def IsEmpty(self):
        return self.isEmpty

    # Get the content of the paint board (return QImage)
    def GetContentAsQImage(self):
        image = self.board.toImage()
        return image

    def paintEvent(self, paintEvent):
        # When drawing, it must use an instance of QPainter
        # Drawing is performed between the begin() function and the end() function.
        # The parameter of begin(param) needs to specify the drawing device, that is, where to put the picture
        # drawPixmap is used to draw objects of type QPixmap
        self.painter.begin(self)
        # 0,0 is the coordinates of the starting point of the upper left corner of the drawing
        # self.board is the drawing to be drawn
        self.painter.drawPixmap(0, 0, self.board)
        self.painter.end()

    def mousePressEvent(self, mouseEvent):
        # When the mouse is pressed, get the current position of the mouse and save it as the last position
        self.currentPos = mouseEvent.pos()
        self.lastPos = self.currentPos


    def mouseMoveEvent(self, mouseEvent):
        # When the mouse moves, update the current position and draw a line
        # between the previous position and the current position
        self.currentPos = mouseEvent.pos()
        self.painter.begin(self.board)

        if self.eraserMode == False:
            # Non-eraser mode
            # Set the brush color and thickness
            self.painter.setPen(QPen(self.penColor, self.thickness))
        else:
            # Eraser mode
            # The brush is pure white with a thickness of 80
            self.painter.setPen(QPen(Qt.white, 80))

        # Draw a line
        self.painter.drawLine(self.lastPos, self.currentPos)
        self.painter.end()
        self.lastPos = self.currentPos

        # Update display
        self.update()

    def mouseReleaseEvent(self, mouseEvent):
        # Paint board is no longer empty
        self.isEmpty = False
