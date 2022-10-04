
#imports
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.uic import loadUiType
import tkinter.filedialog
from PIL import Image
import tomopy
import math
import time
import os
import scipy
from scipy import ndimage
from scipy import stats
from scipy import signal
import qimage2ndarray
import pyqtgraph as pg


Ui_on_the_fly_Window, Q_on_the_fly_Window = loadUiType('on_the_fly_CT_tester.ui')  # GUI vom Hauptfenster

class On_the_fly_CT_tester(Ui_on_the_fly_Window, Q_on_the_fly_Window):


    def __init__(self):
        super(On_the_fly_CT_tester, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('On_the_fly_CT_tester')

        self.pushLoad.clicked.connect(self.load)
        self.pushReconstruct.clicked.connect(self.reconstruct)
        self.COR.valueChanged.connect(self.reconstruct)
        self.brightness.valueChanged.connect(self.reconstruct)
        self.speed_W.valueChanged.connect(self.reconstruct)


        print('on_the_fly_CT_tester.py init')



    def load(self):

        self.pushLoad.setEnabled(False)
        self.slice_number.setEnabled(False)
        self.COR.setEnabled(False)
        self.brightness.setEnabled(False)
        self.speed_W.setEnabled(False)

        path_klick_raw = QtGui.QFileDialog.getOpenFileName(self, 'Select one file of the normalized projections, please.')
        path_klick = path_klick_raw[0]
        print(path_klick)

        htap = path_klick[::-1]
        self.path_in = path_klick[0: len(htap) - htap.find('/') - 1: 1]
        self.namepart = path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 5: 1]
        self.counter = int(path_klick[len(htap) - htap.find('.') - 5: len(htap) - htap.find('.') - 1:1])
        self.filetype = path_klick[len(htap) - htap.find('.') - 1: len(htap):1]
        self.path_out = self.path_in + '/on_the_fly_CT_reco_test'
        if os.path.isdir(self.path_out) is False:
            os.mkdir(self.path_out)

        reference = Image.open(path_klick)
        self.full_size = reference.size[0]
        print(self.full_size)

        # LOADING FROM DISK #
        for i in range(self.counter,999999):
            print(i)
            filename = self.path_in + self.namepart + str(i).zfill(4) + self.filetype

            if i == self.counter:
                self.A = numpy.zeros(reference.size[0], reference.size[1], dtype=float)
            if os.path.exists(filename) != True:
                break

            QtWidgets.QApplication.processEvents()
            print('Loading ', filename)
            proj = Image.open(filename)
            proj_ = numpy.array(proj)
            newrow = proj_[self.slice_number.value(), :]
            self.A = numpy.vstack([self.A, newrow])
        print('A',self.A.shape)
        self.B = numpy.transpose(self.A, (1,0))

        self.pushLoad.setEnabled(True)
        self.pushReconstruct.setEnabled(True)
        self.slice_number.setEnabled(True)
        self.COR.setEnabled(True)
        self.brightness.setEnabled(True)
        self.speed_W.setEnabled(True)



    def reconstruct(self):
        self.pushLoad.setEnabled(False)
        self.pushReconstruct.setEnabled(False)
        self.slice_number.setEnabled(False)
        self.COR.setEnabled(False)
        self.brightness.setEnabled(False)
        self.speed_W.setEnabled(False)

        QtWidgets.QApplication.processEvents()
        print('def reconstruct')

        self.full_size = self.A.shape[1]
        self.number_of_projections = self.A.shape[0]

        new_list = numpy.arange(self.number_of_projections) * self.speed_W.value() * math.pi / 180
        print(new_list)
        print(new_list.shape)

        center_list = [self.COR.value() + round(0.5 * self.full_size)] * (self.number_of_projections)
        print(center_list)
        print(len(center_list))

        sinos = numpy.zeros((2, self.full_size, self.number_of_projections), dtype=float)
        sinos[0,:,:] = self.B * 16000
        sinos[1,:,:] = self.B
        transposed_sinos = numpy.transpose(sinos, axes = [2,0,1])
        print('transposed_sinos_shape', transposed_sinos.shape)

        extended_sinos = tomopy.misc.morph.pad(transposed_sinos, axis=2, npad=round(0.5 * self.full_size), mode='edge')
        extended_sinos = tomopy.minus_log(extended_sinos)
        extended_sinos = numpy.nan_to_num(extended_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)

        slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm='gridrec', filter_name='shepp')
        slices = slices[:,round(0.5 * self.full_size) : -round(0.5 * self.full_size) , round(0.5 * self.full_size) : -round(0.5 * self.full_size)]
        slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)
        original_reconstruction = slices[0, :, :]
        print('reconstructions done')

        """
        img = Image.fromarray(original_reconstruction)
        self.filename_out4 = self.path_out + self.namepart + 'test_reconstruction' + self.filetype
        img.save(self.filename_out4)
        print('reconstruction saved')

        img = Image.fromarray(sinos[0,:,:])
        self.filename_out5 = self.path_out + self.namepart + 'test_sino' + self.filetype
        img.save(self.filename_out5)
        print('sino saved')
        """

        myarray = original_reconstruction * self.brightness.value()  # * contrast - (contrast - 128)  # 2048 - 1920
        myarray = myarray.repeat(2, axis=0).repeat(2, axis=1)
        yourQImage = qimage2ndarray.array2qimage(myarray)
        self.test_reco.setPixmap(QPixmap(yourQImage))

        self.pushLoad.setEnabled(True)
        self.pushReconstruct.setEnabled(True)
        self.slice_number.setEnabled(True)
        self.COR.setEnabled(True)
        self.brightness.setEnabled(True)
        self.speed_W.setEnabled(True)
        print('Done!')


"""
    def apply_to_volume(self):
        self.pushApplyVolume.setEnabled(False)
        self.pushReconstruct.setEnabled(False)
        self.doubleSpinBoxCOR.setEnabled(False)
        self.pushAnalyze.setEnabled(False)
        self.scan_range.setEnabled(False)
        self.slice_number.setEnabled(False)
        self.pushCorrect.setEnabled(False)
        self.filter_list.setEnabled(False)
        self.savgol_window.setEnabled(False)
        self.savgol_poly.setEnabled(False)

        self.movement_corrector_results = self.correction_list

        self.path_out2 = self.path_in + '/movement_corrected_projections'
        if os.path.isdir(self.path_out2) is False:
            os.mkdir(self.path_out2)

        n = 0
        while n < self.number_of_projections:
            self.progressBar.setValue((n + 1) * 100 / self.number_of_projections)
            QtWidgets.QApplication.processEvents()

            filename = self.path_in + self.namepart + str(n+1).zfill(4) + self.filetype
            print('Loading ', filename)
            proj = Image.open(filename)
            proj_ = numpy.array(proj)
            #print('removing nan')
            proj_ = numpy.nan_to_num(proj_, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
            #print('shifting projection')
            corrected = ndimage.shift(proj_, (0, self.correction_list[n]), mode='nearest', prefilter=True)

            self.filename2 = self.path_out2 + self.namepart + str(n + 1).zfill(4) + self.filetype
            img = Image.fromarray(corrected)
            img.save(self.filename2)
            print('Writing ', self.filename2)

            n = n + 1

        self.pushReconstruct.setEnabled(True)
        self.doubleSpinBoxCOR.setEnabled(True)
        self.pushAnalyze.setEnabled(True)
        self.scan_range.setEnabled(True)
        self.slice_number.setEnabled(True)
        self.pushCorrect.setEnabled(True)
        self.filter_list.setEnabled(True)
        self.savgol_window.setEnabled(True)
        self.savgol_poly.setEnabled(True)
        self.pushApplyVolume.setEnabled(True)
"""



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    main = On_the_fly_CT_tester()
    main.show()
    sys.exit(app.exec_())