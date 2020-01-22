import sys
import cv2
import numpy as np
import PySide2.QtWidgets as qtw
import PySide2.QtCore as qtc
import PySide2.QtGui as qtg
import matplotlib.pyplot as plt

from src.model.model_serve import ModelServer
from pathlib import Path

N_PCA = 100


class FPCA(qtw.QWidget):
    def __init__(self, parent=None):
        super(FPCA, self).__init__(parent)
        self.setWindowTitle("FPCA")
        self.maingrid = qtw.QGridLayout(self)

        # Creates 100 sliders
        self.slider_grid = qtw.QWidget()
        slider_grid_layout = qtw.QGridLayout(self.slider_grid)
        slider_grid_layout.setVerticalSpacing(30)

        self.slider_list = []
        for i in range(N_PCA):
            slider = qtw.QSlider(orientation=qtc.Qt.Horizontal)
            slider.sliderReleased.connect(lambda i=i: self.recolor(i))
            slider.setFixedWidth(300)
            slider.setMinimum(-300)
            slider.setMaximum(300)
            slider.setValue(0)
            self.slider_list.append(slider)
            slider_grid_layout.addWidget(qtw.QLabel(f"PC{i + 1}"), i, 0)
            slider_grid_layout.addWidget(slider, i, 1)

        # Places these sliders in a scrollable area
        sliders = qtw.QScrollArea()
        sliders.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOn)
        sliders.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        sliders.setWidget(self.slider_grid)

        # Creates a pixmap of the mean face
        self.imgarr = self.get_mean_face()
        pic = cv2.resize(self.imgarr, (0, 0), fx=10, fy=10)
        height, width, channel = pic.shape
        face_image = qtg.QImage(pic.data, width, height, 3 * width, qtg.QImage.Format_RGB888).rgbSwapped()
        face_pixmap = qtg.QPixmap.fromImage(face_image)
        self.face_label = qtw.QLabel()
        self.face_label.setPixmap(face_pixmap)

        # Creates a grid containing three push-buttons
        self.buttons_grid = qtw.QWidget()
        buttons_grid_layout = qtw.QGridLayout(self.buttons_grid)
        self.randomize_button = qtw.QPushButton("Random")
        self.mean_button = qtw.QPushButton("Mean")
        self.invert_button = qtw.QPushButton("Invert")
        for i, b in enumerate([self.randomize_button, self.mean_button, self.invert_button]):
            buttons_grid_layout.addWidget(b, 0, i)

        # Adds the separate widgets to the main grid
        self.maingrid.addWidget(sliders, 0, 0)
        self.maingrid.addWidget(self.face_label, 0, 1)
        self.maingrid.addWidget(self.buttons_grid, 1, 0)

        self.model_server = ModelServer(Path(__file__).parents[1] / 'model' / 'saved_models' / 'model004.h5', 'famousbirthdays')

        def randomize_action():
            self.model_server.random_person()
            for i, pc in enumerate(self.model_server.pc_coeffs[0]):
                self.slider_list[i].setValue(int(pc * 150))
            self.recolor()

        self.randomize_button.clicked.connect(randomize_action)

    def recolor(self, i=None):
        if i is not None:
            self.model_server.set_pc_coeff(i, self.slider_list[i].value() / 150)
        # self.imgarr = np.full(self.imgarr.shape, int((self.slider_list[0].value() + 300)*255/600), dtype=np.uint8)

        self.model_server.update_image()
        self.imgarr = self.model_server.image.numpy()

        pic = cv2.resize(self.imgarr, (0, 0), fx=10, fy=10)
        print(pic.shape)
        print(pic.nbytes / pic.shape[1])
        plt.imshow(np.array(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)))
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        plt.show()
        face_image = qtg.QImage(pic.data, pic.shape[1], pic.shape[0], pic.shape[1]*3, qtg.QImage.Format_RGBX64).rgbSwapped()
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
