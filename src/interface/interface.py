import sys
import cv2
import numpy as np
import PySide2.QtWidgets as qtw
import PySide2.QtCore as qtc
import PySide2.QtGui as qtg

N_PCA = 100


class FPCA(qtw.QWidget):

    def __init__(self, parent = None):
        super(FPCA, self).__init__(parent)
        self.setWindowTitle("FPCA")
        self.maingrid = qtw.QGridLayout(self)

        # Creates 100 sliders
        loose_sliders = qtw.QWidget()
        box_of_sliders = qtw.QVBoxLayout(loose_sliders)

        self.slider_list = []
        for i in range(N_PCA):
            slider = qtw.QSlider(orientation=qtc.Qt.Horizontal)
            slider.sliderReleased.connect(lambda i=i: self.recolor(i))
            slider.setFixedWidth(300)
            slider.setMinimum(-300)
            slider.setMaximum(300)
            slider.setValue(0)
            self.slider_list.append(slider)
            box_of_sliders.addWidget(slider)

        # Places these sliders in a scrollable area
        sliders = qtw.QScrollArea()
        sliders.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOn)
        sliders.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        sliders.setWidget(loose_sliders)

        # Creates a pixmap of the mean face
        self.imgarr = self.get_mean_face()
        pic = cv2.resize(self.imgarr, (0, 0), fx=10, fy=10)
        height, width, channel = pic.shape
        face_image = qtg.QImage(pic.data, width, height, 3 * width, qtg.QImage.Format_RGB888).rgbSwapped()
        face_pixmap = qtg.QPixmap.fromImage(face_image)
        self.face_label = qtw.QLabel()
        self.face_label.setPixmap(face_pixmap)

        # Adds the separate widgets to the main grid
        self.maingrid.addWidget(sliders, 0, 0)
        self.maingrid.addWidget(self.face_label, 0, 1)

    def recolor(self, i):
        print(self.slider_list[i].value())
        self.imgarr = np.full(self.imgarr.shape, int((self.slider_list[0].value() + 300)*255/600), dtype=np.uint8)
        pic = cv2.resize(self.imgarr, (0, 0), fx=10, fy=10)
        face_image = qtg.QImage(pic.data, 640, 640, 3 * 640, qtg.QImage.Format_RGB888).rgbSwapped()
        face_pixmap = qtg.QPixmap.fromImage(face_image)
        self.face_label.setPixmap(face_pixmap)
        self.face_label.update()

    def get_mean_face(self):
        return cv2.imread("./rob.jpg")


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = FPCA()
    window.show()
    sys.exit(app.exec_())
