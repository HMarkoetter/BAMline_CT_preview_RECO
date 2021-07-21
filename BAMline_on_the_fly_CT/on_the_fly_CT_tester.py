
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
        self.pushReconstruct_all.clicked.connect(self.reconstruct_all)
        self.push_Crop_volume.clicked.connect(self.crop_volume)
        #self.slice_number.valueChanged.connect(self.reconstruct)
        #self.COR.valueChanged.connect(self.reconstruct)
        #self.Offset_Angle.valueChanged.connect(self.reconstruct)
        #self.brightness.valueChanged.connect(self.reconstruct)
        #self.speed_W.valueChanged.connect(self.reconstruct)
        #self.algorithm_list.currentIndexChanged.connect(self.reconstruct)
        #self.filter_list.currentIndexChanged.connect(self.reconstruct)

        self.block_size = 128
        self.extend_FOV = 0.05
        #print('on_the_fly_CT_tester.py init')



    def load(self):

        self.pushLoad.setEnabled(False)
        self.slice_number.setEnabled(False)
        self.COR.setEnabled(False)
        self.brightness.setEnabled(False)
        self.speed_W.setEnabled(False)


        path_klick_FF = QtGui.QFileDialog.getOpenFileName(self, 'Select first FF-file, please.', "C:\Fly_and_Helix_Test\Flying-CT_Test\MEA_Flying-CT_2x2_crop_normalized")
        self.path_klickFF = path_klick_FF[0]
        print(self.path_klickFF)

        htapFF = self.path_klickFF[::-1]
        self.path_inFF = self.path_klickFF[0: len(htapFF) - htapFF.find('/') - 1: 1]
        self.namepartFF = self.path_klickFF[len(htapFF) - htapFF.find('/') - 1: len(htapFF) - htapFF.find('.') - 5: 1]
        self.counterFF = int(self.path_klickFF[len(htapFF) - htapFF.find('.') - 5: len(htapFF) - htapFF.find('.') - 1:1])
        self.filetypeFF = self.path_klickFF[len(htapFF) - htapFF.find('.') - 1: len(htapFF):1]

        referenceFF = Image.open(self.path_klickFF)
        self.full_sizeFF = referenceFF.size[0]
        print(self.full_sizeFF)

        for i in range(self.counterFF,999999):
            print(i)
            lastFF = i
            filenameFF = self.path_inFF + self.namepartFF + str(i).zfill(4) + self.filetypeFF

            if os.path.exists(filenameFF) != True:
                break

        print(lastFF)
        self.FF = numpy.zeros((lastFF - self.counterFF + 1, referenceFF.size[1], referenceFF.size[0]), dtype=float)

        i = self.counterFF
        while i < lastFF:
            # progressBar
            self.progressBar.setValue((i + 1) * 100 / lastFF)

            filenameFF = self.path_inFF + self.namepartFF + str(i).zfill(4) + self.filetypeFF

            QtWidgets.QApplication.processEvents()
            print('LoadingFF ', filenameFF)
            projFF = Image.open(filenameFF)
            proj_FF = numpy.array(projFF)

            # print('A', self.A.shape)
            # print('proj_', proj_.shape)

            self.FF[i - self.counterFF, :, :] = proj_FF

            i = i + 1

        FFavg = numpy.mean(self.FF, axis=0)





        path_klick_raw = QtGui.QFileDialog.getOpenFileName(self, 'Select first projection, please.', "C:\Fly_and_Helix_Test\Flying-CT_Test\MEA_Flying-CT_2x2_crop_normalized")
        self.path_klick = path_klick_raw[0]
        print(self.path_klick)

        htap = self.path_klick[::-1]
        self.path_in = self.path_klick[0: len(htap) - htap.find('/') - 1: 1]
        self.namepart = self.path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 5: 1]
        self.counter = int(self.path_klick[len(htap) - htap.find('.') - 5: len(htap) - htap.find('.') - 1:1])
        self.filetype = self.path_klick[len(htap) - htap.find('.') - 1: len(htap):1]
        #self.path_out = self.path_in + '/on_the_fly_CT_reco_test'
        #if os.path.isdir(self.path_out) is False:
        #    os.mkdir(self.path_out)

        reference = Image.open(self.path_klick)
        self.full_size = reference.size[0]
        print(self.full_size)

        # LOADING FROM DISK #
        # determine last image #
        for i in range(self.counter,999999):
            print(i)
            last = i
            filename = self.path_in + self.namepart + str(i).zfill(4) + self.filetype

            if os.path.exists(filename) != True:
                break

        print(last)
        self.A = numpy.zeros((last - self.counter + 1, reference.size[1], reference.size[0]), dtype=numpy.uint16)

        i = self.counter
        while i < last:

            #progressBar
            self.progressBar.setValue((i + 1) * 100 / last)

            filename = self.path_in + self.namepart + str(i).zfill(4) + self.filetype

            QtWidgets.QApplication.processEvents()
            print('Loading ', filename)
            proj = Image.open(filename)
            proj_ = numpy.array(proj)
            proj_norm = numpy.divide(proj_, FFavg)
            proj_norm = numpy.clip(proj_norm, 0.03, 4)
            proj_norm = proj_norm * 16000
            #print('norm min vs max', numpy.amin(proj_norm), numpy.amax(proj_norm))
            proj_norm = numpy.nan_to_num(proj_norm, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
            #print('norm min vs max', numpy.amin(proj_norm), numpy.amax(proj_norm))
            #proj_norm_16 = proj_norm.astype(numpy.uint16)
            proj_norm_16 = numpy.uint16(proj_norm)
            #print('norm min vs max', numpy.amin(proj_norm_16), numpy.amax(proj_norm_16))
            #print('A', self.A.shape)
            #print('proj_', proj_.shape)

            self.A[i - self.counter,:,:] = proj_norm_16

            i = i + 1


        print('A',self.A.shape, 'A min vs max', numpy.amin(self.A), numpy.amax(self.A))
        #self.A = numpy.transpose(self.A, (0,1,2)) * 16000

        self.slice_number.setMaximum(reference.size[1]-1)
        self.slice_number.setMinimum(0)
        self.slice_number.setValue(round(reference.size[1]/2))

        self.COR.setValue(round(reference.size[0]/2))

        self.spinBox_first.setValue(0)
        self.spinBox_last.setValue(reference.size[1]-1)


        self.pushLoad.setEnabled(True)
        self.pushReconstruct.setEnabled(True)
        self.pushReconstruct_all.setEnabled(True)
        self.slice_number.setEnabled(True)
        self.COR.setEnabled(True)
        self.brightness.setEnabled(True)
        self.Offset_Angle.setEnabled(True)
        self.speed_W.setEnabled(True)
        self.spinBox_first.setEnabled(True)
        self.spinBox_last.setEnabled(True)
        self.push_Crop_volume.setEnabled(True)



    def crop_volume(self):
        self.A = self.A[:,self.spinBox_first.value():self.spinBox_last.value()+1,:]
        self.spinBox_first.setValue(0)
        self.spinBox_last.setValue(self.A.shape[1]-1)
        self.slice_number.setMaximum(self.A.shape[1]-1)
        print('Volume Cropped')
        print('A', self.A.shape)

    def reconstruct(self):
        self.pushLoad.setEnabled(False)
        self.pushReconstruct.setEnabled(False)
        self.pushReconstruct_all.setEnabled(False)
        self.slice_number.setEnabled(False)
        self.COR.setEnabled(False)
        self.brightness.setEnabled(False)
        self.Offset_Angle.setEnabled(False)
        self.speed_W.setEnabled(False)

        QtWidgets.QApplication.processEvents()
        print('def reconstruct')

        self.full_size = self.A.shape[2]
        self.number_of_projections = self.A.shape[0]

        new_list = (numpy.arange(self.number_of_projections) * self.speed_W.value() + self.Offset_Angle.value()) * math.pi / 180
        print(new_list.shape)

        self.extend_FOV = (abs(self.COR.value() - self.A.shape[2]/2))/ (1 * self.A.shape[2]) + 0.05    # extend field of view (FOV), 0.0 no extension, 0.5 half extension to both sides (for half sided 360 degree scan!!!)
        print('extend_FOV ', self.extend_FOV)


        center_list = [self.COR.value() + round(self.extend_FOV * self.full_size)] * (self.number_of_projections)
        print(len(center_list))

        transposed_sinos = numpy.zeros((self.number_of_projections, 1, self.full_size), dtype=float)

        transposed_sinos[:,0,:] = self.A[:, self.slice_number.value(),:]

        print('transposed_sinos_shape', transposed_sinos.shape)

        extended_sinos = tomopy.misc.morph.pad(transposed_sinos, axis=2, npad=round(self.extend_FOV * self.full_size), mode='edge')
        extended_sinos = tomopy.minus_log(extended_sinos)
        extended_sinos = (extended_sinos + 9.68) * 1000  # conversion factor to uint
        extended_sinos = numpy.nan_to_num(extended_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)

        slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm=self.algorithm_list.currentText(),
                                  filter_name=self.filter_list.currentText())
        #slices = slices[:,round(self.extend_FOV * self.full_size) : -round(self.extend_FOV * self.full_size) , round(self.extend_FOV * self.full_size) : -round(self.extend_FOV * self.full_size)]
        slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)
        original_reconstruction = slices[0, :, :]
        print(numpy.amin(original_reconstruction))
        print(numpy.amax(original_reconstruction))
        self.min.setText(str(numpy.amin(original_reconstruction)))
        self.max.setText(str(numpy.amax(original_reconstruction)))
        print('reconstructions done')


        myarray = (original_reconstruction - numpy.amin(original_reconstruction)) * self.brightness.value() / (numpy.amax(original_reconstruction) - numpy.amin(original_reconstruction))
        myarray = myarray.repeat(2, axis=0).repeat(2, axis=1)
        yourQImage = qimage2ndarray.array2qimage(myarray)
        self.test_reco.setPixmap(QPixmap(yourQImage))

        self.pushLoad.setEnabled(True)
        self.pushReconstruct.setEnabled(True)
        self.pushReconstruct_all.setEnabled(True)
        self.slice_number.setEnabled(True)
        self.COR.setEnabled(True)
        self.Offset_Angle.setEnabled(True)
        self.brightness.setEnabled(True)
        self.speed_W.setEnabled(True)
        print('Done!')


    def reconstruct_all(self):
        self.pushLoad.setEnabled(False)
        self.pushReconstruct.setEnabled(False)
        self.slice_number.setEnabled(False)
        self.COR.setEnabled(False)
        self.brightness.setEnabled(False)
        self.Offset_Angle.setEnabled(False)
        self.speed_W.setEnabled(False)

        QtWidgets.QApplication.processEvents()
        print('def reconstruct complete volume')

        self.path_out_reconstructed_full = QtGui.QFileDialog.getExistingDirectory(self, 'Select folder fpr reconstructions.', self.path_klick)

        self.full_size = self.A.shape[2]
        self.number_of_projections = self.A.shape[0]
        print('X-size', self.A.shape[2])
        print('Nr of projections', self.A.shape[0])
        print('Nr of slices', self.A.shape[1])



        new_list = (numpy.arange(self.number_of_projections) * self.speed_W.value() + self.Offset_Angle.value()) * math.pi / 180
        #print(new_list)
        print(new_list.shape)

        self.extend_FOV = (abs(self.COR.value() - self.A.shape[2]/2))/ (1 * self.A.shape[2]) + 0.05    # extend field of view (FOV), 0.0 no extension, 0.5 half extension to both sides (for half sided 360 degree scan!!!)
        print('extend_FOV ', self.extend_FOV)

        center_list = [self.COR.value() + round(self.extend_FOV * self.full_size)] * (self.number_of_projections)
        #print(center_list)
        print(len(center_list))



        i = 0
        while (i < math.ceil(self.A.shape[1] / self.block_size)):

            print('Reconstructing block', i + 1, 'of', math.ceil(self.A.shape[1] / self.block_size))

            extended_sinos = self.A[:, i * self.block_size: (i + 1) * self.block_size, :]
            extended_sinos = tomopy.misc.morph.pad(extended_sinos, axis=2, npad=round(self.extend_FOV * self.full_size), mode='edge')
            extended_sinos = tomopy.minus_log(extended_sinos)
            extended_sinos = (extended_sinos + 9.68) * 1000  # conversion factor to uint

            extended_sinos = numpy.nan_to_num(extended_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)

            slices = tomopy.recon(extended_sinos, new_list,
                                  center=center_list, algorithm=self.algorithm_list.currentText(),
                                  filter_name=self.filter_list.currentText())
            #slices = slices[:, round(self.extend_FOV * self.full_size): -round(self.extend_FOV * self.full_size), round(self.extend_FOV * self.full_size): -round(self.extend_FOV * self.full_size)]
            slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)

            if self.radioButton_16bit_integer.isChecked() == True:
                ima3 = 65535 * (slices - self.int_low.value()) / (self.int_high.value() - self.int_low.value())
                ima3 = numpy.clip(ima3, 1, 65534)
                slices_save = ima3.astype(numpy.uint16)

            if self.radioButton_32bit_float.isChecked() == True:
                slices_save = slices

            print('Reconstructed Volume is', slices_save.shape)

            a = 1
            while (a < self.block_size + 1) and (a < slices_save.shape[0] + 1):

                self.progressBar.setValue((a + (i * self.block_size)) * 100 / self.A.shape[1])

                filename2 = self.path_out_reconstructed_full + self.namepart + str(a + i * self.block_size).zfill(4) + self.filetype
                print('Writing Reconstructed Slices:', filename2)
                slice_save = slices_save[a - 1, :, :]
                img = Image.fromarray(slice_save)
                img.save(filename2)
                QtCore.QCoreApplication.processEvents()
                time.sleep(0.02)

                a = a + 1

            i = i + 1

        self.pushLoad.setEnabled(True)
        self.pushReconstruct.setEnabled(True)
        self.slice_number.setEnabled(True)
        self.COR.setEnabled(True)
        self.brightness.setEnabled(True)
        self.Offset_Angle.setEnabled(True)
        self.speed_W.setEnabled(True)
        print('Done!')



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    main = On_the_fly_CT_tester()
    main.show()
    sys.exit(app.exec_())