import cv2
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import numpy as np
from matplotlib import pyplot as plt  # Import matplotlib


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showimage.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.grayButton.clicked.connect(self.grayClicked)
        self.actionBrightness.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.contrastStretching)
        self.actionNegative_Image.triggered.connect(self.negativeImage)
        self.actionBiner_Image.triggered.connect(self.binaryImage)
        self.actionGray_Histogram.triggered.connect(self.histogram)
        self.actionRGB_Histogram.triggered.connect(self.RGBHistogram)
        self.actionEqual_Histogram.triggered.connect(self.EqualHistogramClicked)
        self.actionTranslasi.triggered.connect(self.translasi)

        # Connect rotation actions
        self.actionRotasi_Minus_45.triggered.connect(lambda: self.rotasi(-45))
        self.actionRotasi_45.triggered.connect(lambda: self.rotasi(45))
        self.actionRotasi_Minus_90.triggered.connect(lambda: self.rotasi(-90))
        self.actionRotasi_90.triggered.connect(lambda: self.rotasi(90))
        self.actionRotasi_180.triggered.connect(lambda: self.rotasi(180))

        self.actionZoom_In.triggered.connect(self.zoomIn)
        self.actionZoom_Out.triggered.connect(self.zoomOut)
        self.actionSkewed_Image.triggered.connect(self.skewedImage)
        self.actionCrop.triggered.connect(self.cropImage)

        # Connect Sobel edge detection action
        self.actionSobel.triggered.connect(self.sobelClicked)

        self.contrast_value = 1.6

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('gambar.jpg')

    @pyqtSlot()
    def grayClicked(self):
        try:
            if self.image is not None:
                H, W = self.image.shape[:2]
                gray = np.zeros((H, W), np.uint8)
                for i in range(H):
                    for j in range(W):
                        gray[i, j] = np.clip(
                            0.299 * self.image[i, j, 0] + 0.587 * self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0,
                            255)
                self.image = gray
                self.displayImage(windows=2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during grayscale conversion: {str(e)}")

    @pyqtSlot()
    def brightness(self):
        try:
            if self.image is not None:
                brightness = 80
                self.image = np.clip(self.image.astype(int) + brightness, 0, 255).astype(np.uint8)
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def contrast(self):
        try:
            if self.image is not None:
                # Apply contrast enhancement
                self.image = np.clip(self.image.astype(float) * self.contrast_value, 0, 255).astype(np.uint8)
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def contrastStretching(self):
        try:
            if self.image is not None:
                min_val = np.min(self.image)
                max_val = np.max(self.image)
                stretched_image = 255 * ((self.image - min_val) / (max_val - min_val))
                self.image = stretched_image.astype(np.uint8)
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def negativeImage(self):
        try:
            if self.image is not None:
                self.image = 255 - self.image
                self.displayImage(1)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during negative transformation: {str(e)}")

    @pyqtSlot()
    def binaryImage(self):
        try:
            if self.image is not None:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
                self.image = binary_image
                self.displayImage(1)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during binary transformation: {str(e)}")

    @pyqtSlot()
    def histogram(self):
        try:
            if self.image is not None:
                if len(self.image.shape) == 3:
                    gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = self.image

                self.image = gray_image
                self.displayImage(2)

                # Plot histogram
                plt.hist(gray_image.ravel(), 255, [0, 255])
                plt.title('Histogram of Grayscale Image')
                plt.xlabel('Pixel Values')
                plt.ylabel('Frequency')
                plt.show()
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during histogram plotting: {str(e)}")

    @pyqtSlot()
    def RGBHistogram(self):
        try:
            if self.image is not None:
                color = ('b', 'g', 'r')
                for i, col in enumerate(color):
                    histo = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                    plt.plot(histo, color=col)
                plt.xlim([0, 256])
                plt.title('Histogram of RGB Image')
                plt.xlabel('Pixel Values')
                plt.ylabel('Frequency')
                plt.show()
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during RGB histogram plotting: {str(e)}")

    @pyqtSlot()
    def EqualHistogramClicked(self):
        try:
            if self.image is not None:
                hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
                cdf = hist.cumsum()

                cdf_normalized = cdf * hist.max() / cdf.max()
                cdf_m = np.ma.masked_equal(cdf, 0)
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                cdf = np.ma.filled(cdf_m, 0).astype('uint8')
                self.image = cdf[self.image]
                self.displayImage(2)

                plt.plot(cdf_normalized, color='b')
                plt.hist(self.image.flatten(), 256, [0, 256], color='r')
                plt.xlim([0, 256])
                plt.legend(('cdf', 'histogram'), loc='upper left')
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during histogram equalization: {str(e)}")

    @pyqtSlot()
    def translasi(self):
        try:
            if self.image is not None:
                h, w = self.image.shape[:2]
                quarter_h, quarter_w = h / 4, w / 4
                T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
                img_translated = cv2.warpAffine(self.image, T, (w, h))
                self.image = img_translated
                self.displayImage(2)  # Display the translated image in the second label
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def rotasi(self, degree):
        try:
            if self.image is not None:
                h, w = self.image.shape[:2]
                rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 0.7)
                cos = np.abs(rotationMatrix[0, 0])
                sin = np.abs(rotationMatrix[0, 1])
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))
                rotationMatrix[0, 2] += (nW / 2) - w / 2
                rotationMatrix[1, 2] += (nH / 2) - h / 2
                rot_image = cv2.warpAffine(self.image, rotationMatrix, (nW, nH))
                self.image = rot_image
                self.displayImage(2)  # Display the rotated image in the second label
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def zoomIn(self):
        try:
            if self.image is not None:
                h, w = self.image.shape[:2]
                resized_img = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Zoomed In Image", resized_img)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def zoomOut(self):
        try:
            if self.image is not None:
                resized_img = cv2.resize(self.image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imshow("Zoomed Out Image", resized_img)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def skewedImage(self):
        try:
            if self.image is not None:
                resized_img = cv2.resize(self.image, (900, 400), interpolation=cv2.INTER_AREA)
                self.image = resized_img
                cv2.imshow("Skewed Image", resized_img)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def cropImage(self):
        try:
            if self.image is not None:
                start_row, start_col = 100, 100
                end_row, end_col = 300, 300

                cropped_img = self.image[start_row:end_row, start_col:end_col]

                cv2.imshow("Cropped Image", cropped_img)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def sobelClicked(self):
        try:
            if self.image is not None:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

                Gx = cv2.filter2D(gray_image, -1, sobel_x)
                Gy = cv2.filter2D(gray_image, -1, sobel_y)

                gradient = np.sqrt(Gx ** 2 + Gy ** 2)
                gradient = (gradient / gradient.max()) * 255

                self.image = gradient.astype(np.uint8)
                self.displayImage(2)  # Display the Sobel edge detected image

                plt.imshow(self.image, cmap='gray', interpolation='bicubic')
                plt.title('Sobel Edge Detection')
                plt.axis('off')
                plt.show()
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during Sobel edge detection: {str(e)}")

    def loadImage(self, flname):
        try:
            self.image = cv2.imread(flname)
            if self.image is None:
                raise FileNotFoundError(f"Could not load image: {flname}")
            self.displayImage()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def displayImage(self, windows=1):
        if self.image is not None:
            qformat = QImage.Format_Indexed8

            if len(self.image.shape) == 3:
                if self.image.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
            img = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)

            img = img.rgbSwapped()
            if windows == 1:
                # Storing the loaded image in imgLabel
                self.imgLabel.setPixmap(QPixmap.fromImage(img))

                # Positioning the image at the center
                self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

            if windows == 2:
                self.hasilLabel.setPixmap(QPixmap.fromImage(img))
                self.hasilLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.hasilLabel.setScaledContents(True)
        else:
            QMessageBox.critical(self, "Error", "No image loaded.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('Show Image GUI')
    window.show()
    sys.exit(app.exec_())
