import cv2
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import numpy as np

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showimage2.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.actionMedianFilter.triggered.connect(self.applyMedianFilter)

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('noisy_image.jpg')  # Load the image with noise

    @pyqtSlot()
    def applyMedianFilter(self):
        try:
            if self.image is not None:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                median_filtered_image = self.median_filter(gray_image)
                self.image = median_filtered_image
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def loadImage(self, fname):
        try:
            self.image = cv2.imread(fname)
            self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def displayImage(self, windows=1):
        try:
            qformat = QImage.Format_Indexed8

            if len(self.image.shape) == 3:
                if self.image.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888

            img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
            img = img.rgbSwapped()

            if windows == 1:
                self.imgLabel.setPixmap(QPixmap.fromImage(img))
                self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.imgLabel.setScaledContents(True)

            if windows == 2:
                self.imgLabel.setPixmap(QPixmap.fromImage(img))
                self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.imgLabel.setScaledContents(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def median_filter(self, image):
        h, w = image.shape
        img_out = image.copy()
        for i in range(3, h - 3):
            for j in range(3, w - 3):
                neighbors = []
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        neighbors.append(image[i + k, j + l])
                neighbors.sort()
                median = neighbors[24]  # Median value in the 49 sorted elements
                img_out[i, j] = median
        return img_out

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('Praktikum Pengolahan Citra Digital')
    window.show()
    sys.exit(app.exec_())
