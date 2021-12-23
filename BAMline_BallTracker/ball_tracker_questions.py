# Ball_Tracker_Questions
# asks for data that needs to be set before ball tracking

from PyQt5 import QtCore, QtGui, QtWidgets
import qimage2ndarray
from PyQt5.uic import loadUiType
import numpy
from PIL import Image
import os
import time
import tkinter.filedialog
from PyQt5.QtGui import QIcon, QPixmap
from scipy import ndimage
import sys

print('We are in Ball_Tracker_Question.py now.')

Ui_Ball_Tracker_Question_Window, Q_Ball_Tracker_Question_Window = loadUiType('Ball_Tracker_Question.ui')  # GUI vom Hauptfenster

class Ball_Tracker_Question(Ui_Ball_Tracker_Question_Window, Q_Ball_Tracker_Question_Window):

    def __init__(self):
        super(Ball_Tracker_Question, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.save_values)
        #print('ball_tracker_questions.py init')

    def save_values(self):
        self.startNumberProj = self.startNumberProj.value()
        self.numberProj = self.numberProj.value()
        self.startNumberFFs = self.startNumberFFs.value()
        self.numberFFs = self.numberFFs.value()
        self.Threshold = self.Threshold.value()
        self.Binning = self.Binning.value()
        self.skip = self.skip.value()
        self.CropTop = self.CropTop.value()
        self.CropBottom = self.CropBottom.value()
        self.CropLeft = self.CropLeft.value()
        self.CropRight = self.CropRight.value()

        self.close()
