# On-the-fly-CT Tester
version =  "Version 2021.11.02 a"

#imports
import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUiType
from PIL import Image
import tomopy
import math
import time
import os
import csv
import qimage2ndarray
from scipy.ndimage.filters import gaussian_filter, median_filter


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
        self.crop_offset = 0

    #def load_hdf5(self, path):


    def load(self):

        self.pushLoad.setEnabled(False)
        self.slice_number.setEnabled(False)
        self.COR.setEnabled(False)
        self.brightness.setEnabled(False)
        self.speed_W.setEnabled(False)
        #self.spinBox_ringradius.setEnabled(False)
        #self.spinBox_DF.setEnabled(False)


        path_klick_FF = QtWidgets.QFileDialog.getOpenFileName(self, 'Select first FF-file, please.', "C:\Fly_and_Helix_Test\Flying-CT_Test\MEA_Flying-CT_2x2_crop_normalized")
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
        FFavg_df = FFavg - self.spinBox_DF.value()


        path_klick_raw = QtWidgets.QFileDialog.getOpenFileName(self, 'Select first projection, please.', self.path_klickFF)
        self.path_klick = path_klick_raw[0]
        print(self.path_klick)

        htap = self.path_klick[::-1]
        self.path_in = self.path_klick[0: len(htap) - htap.find('/') - 1: 1]
        ni_htap = self.path_in[::-1]
        self.folder_name = self.path_klick[len(htap) - htap.find('/') - ni_htap.find('/') -1 :len(htap) - htap.find('/') - 1 : 1]
        self.namepart = self.path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 5: 1]
        self.counter = int(self.path_klick[len(htap) - htap.find('.') - 5: len(htap) - htap.find('.') - 1:1])
        self.filetype = self.path_klick[len(htap) - htap.find('.') - 1: len(htap):1]
        #print(self.folder_name)
        self.Sample.setText(self.path_in)

        reference = Image.open(self.path_klick)
        self.full_size = reference.size[0]
        print(self.full_size)



        # LOADING FROM DISK #
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

            self.progressBar.setValue((i + 1) * 100 / last)

            filename = self.path_in + self.namepart + str(i).zfill(4) + self.filetype

            QtWidgets.QApplication.processEvents()
            print('Loading ', filename)
            proj = Image.open(filename)
            proj_ = numpy.array(proj)
            proj_df = proj_ - self.spinBox_DF.value()
            proj_norm = numpy.divide(proj_df, FFavg_df)
            proj_norm = numpy.clip(proj_norm, 0.03, 4)
            proj_norm = proj_norm * 16000
            proj_norm = numpy.nan_to_num(proj_norm, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
            proj_norm_16 = numpy.uint16(proj_norm)

            self.A[i - self.counter,:,:] = proj_norm_16

            if self.spinBox_ringradius.value() != 0:

                #print('Radius for Ring reduction: ', self.spinBox_ringradius.value())

                if i == self.counter:
                    #print('i == self.counter')
                    self.proj_sum = numpy.array(proj_norm_16, dtype=numpy.single)
                    #self.proj_sum = proj_norm_16
                else:
                    #print('i != self.counter')
                    self.proj_sum = self.proj_sum + proj_norm_16

            i = i + 1

        print('A', self.A.shape, 'A min vs max', numpy.amin(self.A), numpy.amax(self.A))

        if self.spinBox_ringradius.value() != 0:
            print('proj_sum dimensions', self.proj_sum.shape)
            proj_sum_filtered = median_filter(self.proj_sum, size = self.spinBox_ringradius.value())
            correction_map = numpy.divide(self.proj_sum, proj_sum_filtered)
            correction_map = numpy.clip(correction_map, 0.9, 1.1)

            print('correction_map', correction_map.shape, 'correction_map min vs max', numpy.amin(correction_map), numpy.amax(correction_map))


            i=0
            while i < self.A.shape[0]:
                self.A[i, :, :] = numpy.uint16(numpy.divide(self.A[i, :, :], correction_map))
                self.progressBar.setValue((i + 1) * 100 / self.A.shape[0])
                QtWidgets.QApplication.processEvents()
                i = i+1


        print('A',self.A.shape, 'A min vs max', numpy.amin(self.A), numpy.amax(self.A))

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
        self.crop_offset = self.crop_offset + self.spinBox_first.value()
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

        self.extend_FOV = 2* (abs(self.COR.value() - self.A.shape[2]/2))/ (1 * self.A.shape[2]) + 0.05    # extend field of view (FOV), 0.0 no extension, 0.5 half extension to both sides (for half sided 360 degree scan!!!)
        print('extend_FOV ', self.extend_FOV)


        if self.number_of_projections * self.speed_W.value() >= 270:
            self.number_of_used_projections = round(360 / self.speed_W.value())
        else:
            print('smaller than 3/2 Pi')
            self.number_of_used_projections = round(180 / self.speed_W.value())
        print('number of used projections', self.number_of_used_projections)

        new_list = (numpy.arange(self.number_of_used_projections) * self.speed_W.value() + self.Offset_Angle.value()) * math.pi / 180
        print(new_list.shape)

        center_list = [self.COR.value() + round(self.extend_FOV * self.full_size)] * (self.number_of_used_projections)
        print(len(center_list))

        transposed_sinos = numpy.zeros((min(self.number_of_used_projections, self.A.shape[0]), 1, self.full_size), dtype=float)
        transposed_sinos[:,0,:] = self.A[0:min(self.number_of_used_projections, self.A.shape[0]), self.slice_number.value(),:]
        print('transposed_sinos_shape', transposed_sinos.shape)

        extended_sinos = tomopy.misc.morph.pad(transposed_sinos, axis=2, npad=round(self.extend_FOV * self.full_size), mode='edge')
        extended_sinos = tomopy.minus_log(extended_sinos)
        extended_sinos = (extended_sinos + 9.68) * 1000  # conversion factor to uint
        extended_sinos = numpy.nan_to_num(extended_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
        if self.checkBox_phase_2.isChecked() == True:
            extended_sinos = tomopy.prep.phase.retrieve_phase(extended_sinos, pixel_size=0.0001, dist=self.doubleSpinBox_distance_2.value(), energy=self.doubleSpinBox_Energy_2.value(), alpha=self.doubleSpinBox_alpha_2.value(), pad=True, ncore=None, nchunk=None)

        if self.algorithm_list.currentText() == 'FBP_CUDA':
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
            slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm=tomopy.astra, options=options)
        else:
            slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm=self.algorithm_list.currentText(),
                                  filter_name=self.filter_list.currentText())

        slices = slices[:,round(self.extend_FOV * self.full_size /2) : -round(self.extend_FOV * self.full_size /2) , round(self.extend_FOV * self.full_size /2) : -round(self.extend_FOV * self.full_size /2)]
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

        self.path_out_reconstructed_ask = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder for reconstructions.', self.path_klick)
        self.path_out_reconstructed_full = self.path_out_reconstructed_ask + '/'+ self.folder_name
        os.mkdir(self.path_out_reconstructed_full)

        self.full_size = self.A.shape[2]
        self.number_of_projections = self.A.shape[0]
        print('X-size', self.A.shape[2])
        print('Nr of projections', self.A.shape[0])
        print('Nr of slices', self.A.shape[1])

        self.extend_FOV = 2* (abs(self.COR.value() - self.A.shape[2]/2))/ (1 * self.A.shape[2]) + 0.05    # extend field of view (FOV), 0.0 no extension, 0.5 half extension to both sides (for half sided 360 degree scan!!!)
        print('extend_FOV ', self.extend_FOV)


        if self.number_of_projections * self.speed_W.value() >= 270:
            self.number_of_used_projections = round(360 / self.speed_W.value())
        else:
            print('smaller than 3/2 Pi')
            self.number_of_used_projections = round(180 / self.speed_W.value())
        print('number of used projections', self.number_of_used_projections)

        new_list = (numpy.arange(self.number_of_used_projections) * self.speed_W.value() + self.Offset_Angle.value()) * math.pi / 180
        print(new_list.shape)

        center_list = [self.COR.value() + round(self.extend_FOV * self.full_size)] * (self.number_of_used_projections)
        print(len(center_list))

        file_name_parameter = self.path_out_reconstructed_full + '/parameter.csv'
        with open(file_name_parameter, mode = 'w', newline='') as parameter_file:
            csv_writer = csv.writer(parameter_file, delimiter = ' ', quotechar=' ')
            csv_writer.writerow(['Path input                    ', self.path_in,' '])
            csv_writer.writerow(['Path output                   ', self.path_out_reconstructed_full,' '])
            csv_writer.writerow(['Number of used projections    ', str(self.number_of_used_projections),' '])
            csv_writer.writerow(['Center of rotation            ', str(self.COR.value()), ' '])
            csv_writer.writerow(['Dark field value              ', str(self.spinBox_DF.value()),' '])
            csv_writer.writerow(['Ring handling radius          ', str(self.spinBox_ringradius.value()),' '])
            csv_writer.writerow(['Rotation offset               ', str(self.Offset_Angle.value()), ' '])
            csv_writer.writerow(['Rotation speed [°/image]      ', str(self.speed_W.value()), ' '])
            csv_writer.writerow(['Phase retrieval               ', str(self.checkBox_phase_2.isChecked()), ' '])
            csv_writer.writerow(['Phase retrieval distance      ', str(self.doubleSpinBox_distance_2.value()), ' '])
            csv_writer.writerow(['Phase retrieval energy        ', str(self.doubleSpinBox_Energy_2.value()), ' '])
            csv_writer.writerow(['Phase retrieval alpha         ', str(self.doubleSpinBox_alpha_2.value()), ' '])
            csv_writer.writerow(['16-bit                        ', str(self.radioButton_16bit_integer.isChecked()), ' '])
            csv_writer.writerow(['16-bit integer low            ', str(self.int_low.value()), ' '])
            csv_writer.writerow(['16-bit integer high           ', str(self.int_high.value()), ' '])
            csv_writer.writerow(['Reconstruction algorithm      ', self.algorithm_list.currentText(), ' '])
            csv_writer.writerow(['Reconstruction filter         ', self.filter_list.currentText(), ' '])
            csv_writer.writerow(['Software Version              ', version, ' '])
            csv_writer.writerow(['binning                       ', '1x1x1', ' '])


        i = 0
        while (i < math.ceil(self.A.shape[1] / self.block_size)):

            print('Reconstructing block', i + 1, 'of', math.ceil(self.A.shape[1] / self.block_size))

            extended_sinos = self.A[0:min(self.number_of_used_projections, self.A.shape[0]), i * self.block_size: (i + 1) * self.block_size, :]
            extended_sinos = tomopy.misc.morph.pad(extended_sinos, axis=2, npad=round(self.extend_FOV * self.full_size), mode='edge')
            extended_sinos = tomopy.minus_log(extended_sinos)
            extended_sinos = (extended_sinos + 9.68) * 1000  # conversion factor to uint

            extended_sinos = numpy.nan_to_num(extended_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
            if self.checkBox_phase_2.isChecked() == True:
                extended_sinos = tomopy.prep.phase.retrieve_phase(extended_sinos, pixel_size=0.0001, dist=self.doubleSpinBox_distance_2.value(), energy=self.doubleSpinBox_Energy_2.value(), alpha=self.doubleSpinBox_alpha_2.value(), pad=True, ncore=None, nchunk=None)

            if self.algorithm_list.currentText() == 'FBP_CUDA':
                options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
                slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm=tomopy.astra,
                                      options=options)
            else:
                slices = tomopy.recon(extended_sinos, new_list, center=center_list,
                                      algorithm=self.algorithm_list.currentText(),
                                      filter_name=self.filter_list.currentText())

            slices = slices[:, round(self.extend_FOV * self.full_size /2): -round(self.extend_FOV * self.full_size /2), round(self.extend_FOV * self.full_size /2): -round(self.extend_FOV * self.full_size /2)]
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

                filename2 = self.path_out_reconstructed_full + self.namepart + str(a + self.crop_offset + i * self.block_size).zfill(4) + self.filetype
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