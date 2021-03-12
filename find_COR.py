# FIND COR
# reads 1st and 3rd file, flips 3rd file, shifts 3rd file and divides it with the 1st => COR is derived from that

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

print('We are in find_COR now.')

Ui_COR_finderWindow, QCOR_finderWindow = loadUiType('find_COR.ui')  # GUI vom Hauptfenster


class COR_finder(Ui_COR_finderWindow, QCOR_finderWindow):


    def __init__(self, path, path_out, index_COR_1, index_COR_2, FF_index, transpose):
        super(COR_finder, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Find the Center of Rotation')

        self.path_klick = path
        self.path_out = path_out
        self.index_COR_1 = index_COR_1
        self.index_COR_2 = index_COR_2
        self.FF_index = FF_index
        self.transpose = transpose

        self.COR_slider.valueChanged.connect(self.shift_COR)
        #self.CORSpinBox.valueChanged.connect(self.shift_COR)
        self.rotate.valueChanged.connect(self.shift_COR)
        self.pushButton.clicked.connect(self.save_COR)
        self.contrastSlider.valueChanged.connect(self.shift_COR)
        print('find COR init')
        #self.done = False
        self.load_COR()
        self.shift_COR()


    def load_COR(self):


        print(self.path_klick)
        htap = self.path_klick[::-1]
        path_in = self.path_klick[0: len(htap) - htap.find('/') - 1: 1]
        namepart = self.path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 5: 1]
        filetype = self.path_klick[len(htap) - htap.find('.') - 1: len(htap):1]

        filename1 = path_in + namepart + str(self.index_COR_1).zfill(4) + filetype
        filename3 = path_in + namepart + str(self.index_COR_2).zfill(4) + filetype
        filename4 = path_in + namepart + str(self.FF_index).zfill(4) + filetype
        filename5 = path_in + namepart + str(self.FF_index + 1).zfill(4) + filetype
        self.filename_out = self.path_out + namepart + 'find_COR' + filetype

        while os.path.exists(filename5) != True:
            time.sleep(2)
            print('waiting for next file:', filename4)

        im_000deg = Image.open(filename1)
        im_180deg = Image.open(filename3)
        FF = Image.open(filename4)

        if self.transpose == True:
            im_000deg = im_000deg.transpose(Image.TRANSPOSE)
            im_180deg = im_180deg.transpose(Image.TRANSPOSE)
            FF = FF.transpose(Image.TRANSPOSE)

        im = im_000deg
        im_000deg = numpy.single(numpy.array(im_000deg))
        im_180deg = numpy.single(numpy.array(im_180deg))
        FF = numpy.single(numpy.array(FF))

        self.im_000_normalized = numpy.divide(im_000deg, FF)
        im_180_normalized = numpy.divide(im_180deg, FF)
        self.im_180_flipped = numpy.flip(im_180_normalized, axis=1)
        self.im_180_flipped = numpy.nan_to_num(self.im_180_flipped, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
        self.full_size = im.size[0]

        print('find COR load')


    def shift_COR(self):
        i = self.COR_slider.value()/10
        contrast = self.contrastSlider.value()
        self.COR_pos.setText(str((i + self.full_size) / 2))

        self.rotated = ndimage.rotate(self.im_180_flipped, self.rotate.value(), axes= [1,0], reshape=False, output=None, order=3, mode='nearest', cval=0.0, prefilter=True)

        im_180_flipped_shifted = ndimage.shift(numpy.single(numpy.array(self.rotated)), [0,i], order=3, mode='nearest', prefilter=True)
        divided = numpy.divide(im_180_flipped_shifted, self.im_000_normalized, out=numpy.zeros_like(im_180_flipped_shifted), where=self.im_000_normalized != 0)

        print(i, self.full_size)
        if 0 < i < self.full_size:
            divided2 = divided[:, int(i)    : self.full_size]
        elif 0 > i > - self.full_size:
            divided2 = divided[:,           : self.full_size + int(i)]
        else:
            divided2 = divided

        myarray = divided2 * contrast - (contrast - 128)   # 2048 - 1920
        yourQImage = qimage2ndarray.array2qimage(myarray)
        self.divided.setPixmap(QPixmap(yourQImage))
        print('find COR shift')


    def save_COR(self):
        i = self.COR_slider.value()/10
        contrast = self.contrastSlider.value()
        self.COR_pos.setText(str((i + self.full_size) / 2))
        #self.CORSpinBox.setValue((i + self.full_size) / 2)

        self.rotated = ndimage.rotate(self.im_180_flipped, self.rotate.value(), axes=[1, 0], reshape=False, output=None, order=3, mode='nearest', cval=0.0, prefilter=True)
        im_180_flipped_shifted = ndimage.shift(numpy.single(numpy.array(self.im_180_flipped)), [0, i], order=3, mode='nearest', prefilter=True)
        divided = numpy.divide(im_180_flipped_shifted, self.im_000_normalized, out=numpy.zeros_like(im_180_flipped_shifted), where=self.im_000_normalized != 0)

        myarray = divided * contrast - (contrast - 128)
        yourQImage = qimage2ndarray.array2qimage(myarray)
        self.divided.setPixmap(QPixmap(yourQImage))

        print('Writing shifted:', self.filename_out)
        img = Image.fromarray(divided)
        img.save(self.filename_out)

        self.COR = (i + self.full_size) / 2
        self.rotate = self.rotate.value()

        print('find COR save')
        self.close()

