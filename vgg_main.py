from vgg_ui import VGGUI
from PyQt5.QtWidgets import QApplication

import sys


def main():
    app = QApplication(sys.argv)

    mainWidget = VGGUI()  # Create a new main screen
    mainWidget.show()  # Display the main screen
    exit(app.exec_())  # Enter the message loop


if __name__ == '__main__':
    main()

