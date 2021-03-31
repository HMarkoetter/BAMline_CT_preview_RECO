# ask_once
# reads asks for data that needs to be set before reconstruction

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

print('We are in ask_once.py now.')

Ui_ask_once_Window, Q_ask_once_Window = loadUiType('ask_once.ui')  # GUI vom Hauptfenster


class ask_once(Ui_ask_once_Window, Q_ask_once_Window):


    def __init__(self, module_choice):
        super(ask_once, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.save_values)
        self.pushButton_path.clicked.connect(self.set_path)
        print('ask_once.py init')
        self.comboBox.addItems(module_choice)
        self.setWindowTitle('BAMline-CT Start')

        self.standard_path = '/mnt/raid/TOPO/2021/03/30_evaluation/'
        self.textBrowser.setText(self.standard_path)
        self.path_out = self.standard_path


    def set_path(self):

        #self.path_out = tkinter.filedialog.askdirectory(title="Select one file of the scan", initialdir=self.standard_path)
        self.path_out = QtWidgets.QFileDialog.getExistingDirectory()
        print(self.path_out)
        self.path_out
        self.textBrowser.setText(self.path_out)



    def save_values(self):
        self.doubleSpinBox_COR = self.doubleSpinBox_COR.value()
        self.doubleSpinBox_Tilt = self.doubleSpinBox_Tilt.value()
        self.doubleSpinBox_PixelSize = self.doubleSpinBox_PixelSize.value()
        self.block_size = self.block_size.value()
        self.dark_field_value = self.dark_field_value.value()
        self.no_of_cores = self.no_of_cores.value()
        #self.index_COR_1 = self.index_COR_1.value()
        #self.index_COR_2 = self.index_COR_2.value()
        #self.FF_index = self.FF_index.value()
        #self.index_pixel_size_1 = self.index_pixel_size_1.value()
        #self.index_pixel_size_2 = self.index_pixel_size_2.value()

        self.checkBox_save_normalized = self.checkBox_save_normalized.isChecked()
        self.checkBox_notknowCOR = self.checkBox_notknowCOR.isChecked()
        self.checkBox_notknowPixelSize = self.checkBox_notknowPixelSize.isChecked()
        self.checkBox_classic_order = self.checkBox_classic_order.isChecked()
        self.module_text = self.comboBox.currentText()
        self.transpose = self.transpose.isChecked()
        self.find_pixel_size_vertical = self.find_pixel_size_vertical.isChecked()

        #self.checkBox_save_shifted_projections = self.checkBox_save_shifted_projections.isChecked()
        #checkBox_save_shifted_projections

        #self.algorithm_list = self.algorithm_list.currentText()
        #self.filter_list = self.filter_list.currentText()

        self.close()

