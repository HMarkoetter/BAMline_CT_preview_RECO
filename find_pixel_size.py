# FIND pixel size
# reads two files, shifts 2nd file and divides it with the 1st => pixel size is derived from that

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

print('We are in find_pixel_size now.')

Ui_PixelSize_finderWindow, QPixelSize_finderWindow = loadUiType('find_pixel_size.ui')  # GUI vom Hauptfenster


class PixelSize_finder(Ui_PixelSize_finderWindow, QPixelSize_finderWindow):


    def __init__(self, path, path_out, index_pixel_size_1, index_pixel_size_2, FF_index, transpose, find_pixel_size_vertical):
        super(PixelSize_finder, self).__init__()

        self.setWindowTitle('Find the Pixel Size')

        self.path_klick = path
        self.path_out = path_out
        self.index_pixel_size_1 = index_pixel_size_1
        self.index_pixel_size_2 = index_pixel_size_2
        self.FF_index = FF_index
        self.transpose = transpose
        self.find_pixel_size_vertical = find_pixel_size_vertical

        htap = self.path_klick[::-1]
        self.path_in = self.path_klick[0: len(htap) - htap.find('/') - 1: 1]
        self.namepart = self.path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 5: 1]
        self.filetype = self.path_klick[len(htap) - htap.find('.') - 1: len(htap):1]

        file_name_parameter = self.path_in + '/parameter.csv'
        print(file_name_parameter)
        f = open(file_name_parameter, 'r')              # Reading scan-scheme parameter 'max_shift'
        for line in f:
            line = line.strip()
            columns = line.split()
            print(columns[0])

            if str(columns[0]) == 'box_lateral_shift':
                self.max_shift = int(columns[1])
                print(columns[1])
        f.close()

        self.setupUi(self)
        self.COR_pos.setText('place-holder')
        self.COR_slider.valueChanged.connect(self.shift)
        self.pushButton.clicked.connect(self.save)
        self.contrastSlider.valueChanged.connect(self.shift)
        self.crop_horizontal_stripe.stateChanged.connect(self.shift)
        self.crop_vertical_stripe.stateChanged.connect(self.shift)
        self.pixel_size = 0
        print('Pixel_size init')
        self.load()
        self.shift()


    def load(self):


        print('max_shift = ', self.max_shift)

        filename1 = self.path_in + self.namepart + str(self.index_pixel_size_1).zfill(4) + self.filetype
        filename2 = self.path_in + self.namepart + str(self.index_pixel_size_2).zfill(4) + self.filetype
        filename4 = self.path_in + self.namepart + str(self.FF_index).zfill(4) + self.filetype
        self.filename_out = self.path_out + self.namepart + 'find_Pixe_Size' + self.filetype

        while os.path.exists(filename4) != True:
            time.sleep(3)
            print('waiting for next file:', filename4)



        if self.find_pixel_size_vertical == True:
            if self.transpose == True:
                im_unshifted = Image.open(filename1)
                im_shifted = Image.open(filename2)
                FF = Image.open(filename4)
            else:
                im_unshifted = Image.open(filename1)
                im_shifted = Image.open(filename2)
                FF = Image.open(filename4)
                im_unshifted = im_unshifted.transpose(Image.TRANSPOSE)
                im_shifted = im_shifted.transpose(Image.TRANSPOSE)
                FF = FF.transpose(Image.TRANSPOSE)

        else:
            if self.transpose == True:
                im_unshifted = Image.open(filename1)
                im_shifted = Image.open(filename2)
                FF = Image.open(filename4)
                im_unshifted = im_unshifted.transpose(Image.TRANSPOSE)
                im_shifted = im_shifted.transpose(Image.TRANSPOSE)
                FF = FF.transpose(Image.TRANSPOSE)
            else:
                im_unshifted = Image.open(filename1)
                im_shifted = Image.open(filename2)
                FF = Image.open(filename4)





        self.full_size = im_unshifted.size[0]
        self.full_size_y = im_unshifted.size[1]

        im_unshifted = numpy.single(numpy.array(im_unshifted))
        im_shifted = numpy.single(numpy.array(im_shifted))
        FF = numpy.single(numpy.array(FF))

        self.im_unshifted_normalized = numpy.divide(im_unshifted, FF)
        self.im_shifted_normalized = numpy.divide(im_shifted, FF)
        self.im_shifted_normalized = numpy.nan_to_num(self.im_shifted_normalized, copy=True, nan=1.0, posinf=1.0, neginf=1.0)

        im_shifted_normalized_shifted = ndimage.shift(numpy.single(numpy.array(self.im_shifted_normalized)), [0, 0], order=3,
                                                      mode='nearest', prefilter=True)
        divided = numpy.divide(im_shifted_normalized_shifted, self.im_unshifted_normalized,
                               out=numpy.zeros_like(self.im_shifted_normalized), where=self.im_unshifted_normalized != 0)
        myarray = divided * 1000 - 872
        #yourQImage = qimage2ndarray.array2qimage(myarray)
        print('Pixel_size load')


    def shift(self):
        i = self.COR_slider.value()/10 + 0.0001
        contrast = self.contrastSlider.value()
        self.COR_pos.setText(str(abs(round((self.max_shift/i),3))) + 'µm')


        if self.crop_horizontal_stripe.isChecked() == True and self.crop_vertical_stripe.isChecked() == True:
            im_shifted_normalized = self.im_shifted_normalized[round(self.full_size_y*4.5/10):round(self.full_size_y*5.5/10),round(self.full_size*4.5/10):round(self.full_size*5.5/10)]
            im_unshifted_normalized = self.im_unshifted_normalized[round(self.full_size_y*4.5/10):round(self.full_size_y*5.5/10),round(self.full_size*4.5/10):round(self.full_size*5.5/10)]

        elif self.crop_horizontal_stripe.isChecked() != True and self.crop_vertical_stripe.isChecked() == True:
            im_shifted_normalized = self.im_shifted_normalized[:,round(self.full_size*4.5/10):round(self.full_size*5.5/10)]
            im_unshifted_normalized = self.im_unshifted_normalized[:,round(self.full_size*4.5/10):round(self.full_size*5.5/10)]

        elif self.crop_horizontal_stripe.isChecked() == True and self.crop_vertical_stripe.isChecked() != True:
            im_shifted_normalized = self.im_shifted_normalized[round(self.full_size_y*4.5/10):round(self.full_size_y*5.5/10),:]
            im_unshifted_normalized = self.im_unshifted_normalized[round(self.full_size_y*4.5/10):round(self.full_size_y*5.5/10),:]

        else:
            im_shifted_normalized = self.im_shifted_normalized
            im_unshifted_normalized = self.im_unshifted_normalized


        im_shifted_normalized_shifted = ndimage.shift(numpy.single(numpy.array(im_shifted_normalized)), [0,i], order=3, mode='nearest', prefilter=True)
        divided = numpy.divide(im_shifted_normalized_shifted, im_unshifted_normalized, out=numpy.zeros_like(im_shifted_normalized), where=im_unshifted_normalized!=0)




        print(i, im_shifted_normalized.shape[1])
        if 0 < i < im_shifted_normalized.shape[1]:
            divided2 = divided[:, int(i)    : im_shifted_normalized.shape[1]            ]
        elif 0 > i > - im_shifted_normalized.shape[1]:
            divided2 = divided[:,           : im_shifted_normalized.shape[1] + int(i)   ]
        else:
            divided2 = divided


        myarray = divided2 * contrast - (contrast - 128)
        yourQImage = qimage2ndarray.array2qimage(myarray)
        self.divided.setPixmap(QPixmap(yourQImage))
        print('Pixel_size shift')


    def save(self):
        i = self.COR_slider.value()/10 + 0.0001
        contrast = self.contrastSlider.value()
        self.COR_pos.setText(str(abs(round((self.max_shift/i),3))) + 'µm')
        im_shifted_normalized_shifted = ndimage.shift(numpy.single(numpy.array(self.im_shifted_normalized)), [0, i], order=3, mode='nearest', prefilter=True)
        divided = numpy.divide(im_shifted_normalized_shifted, self.im_unshifted_normalized, out=numpy.zeros_like(self.im_shifted_normalized), where=self.im_unshifted_normalized != 0)
        myarray = divided * contrast - (contrast - 128)
        yourQImage = qimage2ndarray.array2qimage(myarray)
        self.divided.setPixmap(QPixmap(yourQImage))
        print('Writing shifted:', self.filename_out)
        img = Image.fromarray(divided)
        img.save(self.filename_out)
        self.pixel_size = abs(round((self.max_shift / i), 3))
        print(self.pixel_size)

        self.close()

