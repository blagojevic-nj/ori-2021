from PyQt5.QtWidgets import QMainWindow, QFileDialog

import numpy as np
import PIL.Image

import sys
from PyQt5.QtWidgets import QApplication
import matplotlib.pyplot as plt

from keras.models import load_model

breeds = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 'bernese_mountain_dog',
          'shih-tzu', 'great_pyrenees', 'pomeranian', 'basenji', 'samoyed']


def breeds_to_serbian(breed):
    if breed == 'scottish_deerhound':
        return "Škotski jelenski hrt"
    elif breed == 'maltese_dog':
        return "Malteški pas"
    elif breed == 'afghan_hound':
        return "Avganistanski hrt"
    elif breed == 'entlebucher':
        return "Entlebuški pastirski pas"
    elif breed == 'bernese_mountain_dog':
        return "Bernski pastirski pas"
    elif breed == 'shih-tzu':
        return "Ši cu"
    elif breed == 'great_pyrenees':
        return "Pirinejski planinski pas"
    elif breed == 'pomeranian':
        return "Pomeranac"
    elif breed == 'basenji':
        return "Basenji"
    elif breed == 'samoyed':
        return "Samojed"


class QImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

    def open(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                   'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        return file_name


if __name__ == '__main__':
    app = QApplication(sys.argv)
    imageViewer = QImageViewer()
    filename = imageViewer.open()
    fullimage = PIL.Image.open(filename)
    image = fullimage.resize((100, 100))
    image = np.array(image)
    image = np.clip(image / 255.0, 0.0, 1.0)
    image = image.reshape((1, 100, 100, 3))
    model = load_model('model.h5')
    rez = model.predict(image)

    plt.imshow(fullimage)
    plt.text(0, -15, "Mislim da je verovatno " + breeds_to_serbian(breeds[np.argmax(rez)]),
             bbox=dict(fill=False, edgecolor='black', linewidth=2), fontsize='large')
    plt.show()
