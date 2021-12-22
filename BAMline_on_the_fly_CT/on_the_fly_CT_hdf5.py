# On-the-fly-CT Reco
version =  "Version 2021.12.22 b"

import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUiType
from PIL import Image
import h5py
import tomopy
import math
import time
import os
import csv
import qimage2ndarray
from scipy.ndimage.filters import gaussian_filter, median_filter


Ui_on_the_fly_Window, Q_on_the_fly_Window = loadUiType('on_the_fly_CT_reco_hdf.ui')  # connect to the GUI for the program

class On_the_fly_CT_tester(Ui_on_the_fly_Window, Q_on_the_fly_Window):


    def __init__(self):
        super(On_the_fly_CT_tester, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('On_the_fly_CT_tester')

        #connect buttons to actions
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

        self.block_size = 16        #volume will be reconstructed blockwise to reduce needed RAM
        self.extend_FOV = 0.05      #the reconstructed area will be enlarged in order to allow off axis scans
        self.crop_offset = 0        #needed for proper volume cropping

    def buttons_deactivate_all(self):
        self.spinBox_ringradius.setEnabled(False)
        self.spinBox_DF.setEnabled(False)
        self.pushLoad.setEnabled(False)

        self.slice_number.setEnabled(False)
        self.COR.setEnabled(False)
        self.Offset_Angle.setEnabled(False)
        self.brightness.setEnabled(False)
        self.speed_W.setEnabled(False)
        self.pushReconstruct.setEnabled(False)
        self.algorithm_list.setEnabled(False)
        self.filter_list.setEnabled(False)

        self.checkBox_phase_2.setEnabled(False)
        self.doubleSpinBox_distance_2.setEnabled(False)
        self.doubleSpinBox_Energy_2.setEnabled(False)
        self.doubleSpinBox_alpha_2.setEnabled(False)

        self.spinBox_first.setEnabled(False)
        self.spinBox_last.setEnabled(False)
        self.push_Crop_volume.setEnabled(False)

        self.pushReconstruct_all.setEnabled(False)
        self.int_low.setEnabled(False)
        self.int_high.setEnabled(False)
    def buttons_activate_load(self):
        self.spinBox_ringradius.setEnabled(True)
        self.spinBox_DF.setEnabled(True)
        self.pushLoad.setEnabled(True)
    def buttons_activate_reco(self):
        self.slice_number.setEnabled(True)
        self.COR.setEnabled(True)
        self.Offset_Angle.setEnabled(True)
        self.brightness.setEnabled(True)
        self.speed_W.setEnabled(True)
        self.pushReconstruct.setEnabled(True)
        self.algorithm_list.setEnabled(True)
        self.filter_list.setEnabled(True)

        self.checkBox_phase_2.setEnabled(True)
        self.doubleSpinBox_distance_2.setEnabled(True)
        self.doubleSpinBox_Energy_2.setEnabled(True)
        self.doubleSpinBox_alpha_2.setEnabled(True)
    def buttons_activate_reco_all(self):
        self.pushReconstruct_all.setEnabled(True)
        self.int_low.setEnabled(True)
        self.int_high.setEnabled(True)
    def buttons_activate_crop_volume(self):
        self.spinBox_first.setEnabled(True)
        self.spinBox_last.setEnabled(True)
        self.push_Crop_volume.setEnabled(True)


    def load(self):

        #grey out the buttons while program is busy
        self.buttons_deactivate_all()

        #ask for first flat field
        path_klick_FF = QtWidgets.QFileDialog.getOpenFileName(self, 'Select first FF-file, please.', "C:\Fly_and_Helix_Test\Flying-CT_Test\MEA_Flying-CT_2x2_crop_normalized")
        self.path_klickFF = path_klick_FF[0]
        print(self.path_klickFF)

        #analyse and cut the path in pieces
        htapFF = self.path_klickFF[::-1]
        self.path_inFF = self.path_klickFF[0: len(htapFF) - htapFF.find('/') - 1: 1]
        self.namepartFF = self.path_klickFF[len(htapFF) - htapFF.find('/') - 1: len(htapFF) - htapFF.find('.') - 5: 1]
        self.counterFF = int(self.path_klickFF[len(htapFF) - htapFF.find('.') - 5: len(htapFF) - htapFF.find('.') - 1:1])
        self.filetypeFF = self.path_klickFF[len(htapFF) - htapFF.find('.') - 1: len(htapFF):1]
        print(self.path_inFF)
        print(self.namepartFF)
        print(self.counterFF)
        print(self.filetypeFF)

        #open file to get the image dimensions
        if self.filetypeFF == '.tif':
            referenceFF = Image.open(self.path_klickFF)
            referenceFF = numpy.array(referenceFF)
            referenceFF = numpy.transpose(referenceFF)
        elif self.filetypeFF == '.h5':
            with h5py.File(self.path_klickFF, 'r') as hdf:
                entry = hdf.get('entry')
                data = entry.get('/entry/data/data')
                referenceFF = numpy.array(data)
                referenceFF = numpy.transpose(referenceFF)
        else:
            print('Error loading file type')

        self.full_sizeFF = referenceFF.shape
        print(self.full_sizeFF)

        #check how many flat fields there are
        for i in range(self.counterFF,999999):
            print(i)
            lastFF = i
            filenameFF = self.path_inFF + self.namepartFF + str(i).zfill(4) + self.filetypeFF

            if os.path.exists(filenameFF) != True:
                break

        print(lastFF)

        #create array to fill FFs in there
        self.FF = numpy.zeros((lastFF - self.counterFF + 1, referenceFF.shape[1], referenceFF.shape[0]), dtype=float)

        #open each flat field and collect in array
        i = self.counterFF
        while i < lastFF:
            self.progressBar.setValue((i + 1) * 100 / lastFF)

            filenameFF = self.path_inFF + self.namepartFF + str(i).zfill(4) + self.filetypeFF

            QtWidgets.QApplication.processEvents()
            print('LoadingFF ', filenameFF)

            if self.filetypeFF == '.tif':
                projFF = Image.open(filenameFF)
                proj_FF = numpy.array(projFF)
            elif self.filetypeFF == '.h5':
                with h5py.File(self.path_klickFF, 'r') as hdf:
                    entry = hdf.get('entry')
                    data = entry.get('/entry/data/data')
                    proj_FF = numpy.array(data)
            else:
                print('Error loading file type')

            self.FF[i - self.counterFF, :, :] = proj_FF

            i = i + 1

        #average all flat fields
        FFavg = numpy.mean(self.FF, axis=0)
        FFavg_df = FFavg - self.spinBox_DF.value()


        #ask for first projection file
        path_klick_raw = QtWidgets.QFileDialog.getOpenFileName(self, 'Select first projection, please.', self.path_klickFF)
        self.path_klick = path_klick_raw[0]
        print(self.path_klick)

        #analyse and cut the path in pieces
        htap = self.path_klick[::-1]
        self.path_in = self.path_klick[0: len(htap) - htap.find('/') - 1: 1]
        ni_htap = self.path_in[::-1]
        self.folder_name = self.path_klick[len(htap) - htap.find('/') - ni_htap.find('/') -1 :len(htap) - htap.find('/') - 1 : 1]
        self.namepart = self.path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 5: 1]
        self.counter = int(self.path_klick[len(htap) - htap.find('.') - 5: len(htap) - htap.find('.') - 1:1])
        self.filetype = self.path_klick[len(htap) - htap.find('.') - 1: len(htap):1]
        self.Sample.setText(self.path_in)

        #open projection file to get dimensions - (probably not necessary)
        if self.filetype == '.tif':
            reference = Image.open(self.path_klick)
            reference = numpy.array(reference)
            reference = numpy.transpose(reference)
        elif self.filetype == '.h5':
            with h5py.File(self.path_klick, 'r') as hdf:
                entry = hdf.get('entry')
                data = entry.get('/entry/data/data')
                reference = numpy.array(data)
                reference = numpy.transpose(reference)
        else:
            print('Error loading file type')

        self.full_size = reference.shape
        print(self.full_size)



        #check how many files there are
        for i in range(self.counter,999999):
            print('finding last image ',i)
            last = i
            filename = self.path_in + self.namepart + str(i).zfill(4) + self.filetype

            if os.path.exists(filename) != True:
                break

        print(last)

        #create array for all projections and for the angles
        self.A = numpy.zeros((last - self.counter + 1, reference.shape[1], reference.shape[0]), dtype=numpy.uint16)
        self.w = numpy.zeros(last - self.counter + 1, dtype=numpy.float64)

        #open each projection, normalize and store in array
        i = self.counter
        while i < last:
            #update progress bar
            self.progressBar.setValue((i + 1) * 100 / last)

            filename = self.path_in + self.namepart + str(i).zfill(4) + self.filetype

            QtWidgets.QApplication.processEvents()
            print('Loading ', filename)

            #open file
            if self.filetype == '.tif':
                proj = Image.open(filename)
                proj_ = numpy.array(proj)
                self.last_zero_proj = 0
            elif self.filetype == '.h5':
                with h5py.File(filename, 'r') as hdf:
                    entry = hdf.get('entry')
                    data = entry.get('/entry/data/data')
                    proj_ = numpy.array(data)
                    w_data = entry.get('/entry/instrument/NDAttributes/CT_MICOS_W')       #get angles from hdf5-file
                    self.w[i - self.counter] = numpy.array(w_data)
                    print('w', self.w[i - self.counter])
                    if round(self.w[i - self.counter]) == 0:                              #notice the last projection at zero degree
                        self.last_zero_proj = i - self.counter + 3                        #assumes 3 images for speeding up the motor
            else:
                print('Error loading file type')

            #normalize
            proj_df = proj_ - self.spinBox_DF.value()
            proj_norm = numpy.divide(proj_df, FFavg_df)
            proj_norm = numpy.clip(proj_norm, 0.03, 4)
            proj_norm = proj_norm * 16000
            proj_norm = numpy.nan_to_num(proj_norm, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
            proj_norm_16 = numpy.uint16(proj_norm)

            #store in array
            self.A[i - self.counter,:,:] = proj_norm_16

            #collect sum of all normalized projections for ring filtering
            if self.spinBox_ringradius.value() != 0:

                if i == self.counter:
                    self.proj_sum = numpy.array(proj_norm_16, dtype=numpy.single)
                else:
                    self.proj_sum = self.proj_sum + proj_norm_16

            i = i + 1

        print('A', self.A.shape, 'A min vs max', numpy.amin(self.A), numpy.amax(self.A))



        #Ring artifact handling
        if self.spinBox_ringradius.value() != 0:
            print('proj_sum dimensions', self.proj_sum.shape)

            #correction map = (sum of projections) / median filter of (sum of projections)
            proj_sum_filtered = median_filter(self.proj_sum, size = self.spinBox_ringradius.value())
            correction_map = numpy.divide(self.proj_sum, proj_sum_filtered)
            correction_map = numpy.clip(correction_map, 0.9, 1.1)

            print('correction_map dimensions', correction_map.shape, 'correction_map min vs max', numpy.amin(correction_map), numpy.amax(correction_map))

            #apply correction map for ring handling
            i=0
            while i < self.A.shape[0]:
                self.A[i, :, :] = numpy.uint16(numpy.divide(self.A[i, :, :], correction_map))
                self.progressBar.setValue((i + 1) * 100 / self.A.shape[0])
                QtWidgets.QApplication.processEvents()
                i = i+1


        print('A dimensions',self.A.shape, 'A min vs max', numpy.amin(self.A), numpy.amax(self.A))

        #prefill slice number, COR, cropping
        self.slice_number.setMaximum(reference.shape[1]-1)
        self.slice_number.setMinimum(0)
        self.slice_number.setValue(round(reference.shape[1]/2))
        self.COR.setValue(round(reference.shape[0]/2))
        self.spinBox_first.setValue(0)
        self.spinBox_last.setValue(reference.shape[1]-1)

        #prefill rotation-speed[°/img]
        print('found angular values for ',self.w.shape, 'projections')
        print(self.w)
        poly_coeff =  numpy.polyfit(numpy.arange(len(self.w[round((last - self.counter + 1) /4) : round((last - self.counter + 1) * 3/4) ])), self.w[round((last - self.counter + 1) /4) : round((last - self.counter + 1) * 3/4) ], 1, rcond=None, full=False, w=None, cov=False)
        print('Plynom coefficients',poly_coeff)
        self.speed_W.setValue(poly_coeff[0])
        print('Last projection at 0 degree/still speeding up: number', self.last_zero_proj)


        #ungrey the buttons for further use of the program
        self.buttons_activate_load()
        self.buttons_activate_reco()
        #self.buttons_activate_crop_volume()
        #self.buttons_activate_reco_all()
        print('Loading/Normalizing complete!')


    def crop_volume(self):
        #grey out the buttons while program is busy
        self.buttons_deactivate_all()

        #crop the top and/or bottom of the data. store the offset value for saving slices under the right number
        self.A = self.A[:,self.spinBox_first.value():self.spinBox_last.value()+1,:]
        self.crop_offset = self.crop_offset + self.spinBox_first.value()
        self.spinBox_first.setValue(0)
        self.spinBox_last.setValue(self.A.shape[1]-1)
        self.slice_number.setMaximum(self.A.shape[1]-1)
        print('Volume Cropped')
        print('A', self.A.shape)

        #ungrey the buttons for further use of the program
        self.buttons_activate_load()
        self.buttons_activate_reco()
        self.buttons_activate_crop_volume()
        self.buttons_activate_reco_all()


    def reconstruct(self):

        # grey out the buttons while program is busy
        self.buttons_deactivate_all()

        QtWidgets.QApplication.processEvents()
        print('def reconstruct')

        self.full_size = self.A.shape[2]
        self.number_of_projections = self.A.shape[0]

        #determine how far to extend field of view (FOV), 0.0 no extension, 0.5 half extension to both sides (for off center 360 degree scan!!!)
        self.extend_FOV = 2* (abs(self.COR.value() - self.A.shape[2]/2))/ (1 * self.A.shape[2]) + 0.05
        print('extend_FOV ', self.extend_FOV)

        #check if the scan was 180° or 360°
        if self.number_of_projections * self.speed_W.value() >= 270:
            self.number_of_used_projections = round(360 / self.speed_W.value())
        else:
            print('smaller than 3/2 Pi')
            self.number_of_used_projections = round(180 / self.speed_W.value())
        print('number of used projections', self.number_of_used_projections)

        #create list with x-positions of projections
        new_list = (numpy.arange(self.number_of_used_projections) * self.speed_W.value() + self.Offset_Angle.value()) * math.pi / 180
        print('x_list ',new_list.shape)

        #create list with all projection angles
        center_list = [self.COR.value() + round(self.extend_FOV * self.full_size)] * (self.number_of_used_projections)
        print('COR_list ',len(center_list))

        #create one sinogram in the form [z, y, x]
        transposed_sinos = numpy.zeros((min(self.number_of_used_projections, self.A.shape[0]), 1, self.full_size), dtype=float)
        #transposed_sinos[:,0,:] = self.A[0:min(self.number_of_used_projections, self.A.shape[0]), self.slice_number.value(),:]
        transposed_sinos[:,0,:] = self.A[self.last_zero_proj : min(self.last_zero_proj + self.number_of_used_projections, self.A.shape[0]), self.slice_number.value(),:]
        print('transposed_sinos_shape', transposed_sinos.shape)

        #extend data with calculated parameter, compute logarithm, remove NaN-values
        extended_sinos = tomopy.misc.morph.pad(transposed_sinos, axis=2, npad=round(self.extend_FOV * self.full_size), mode='edge')
        extended_sinos = tomopy.minus_log(extended_sinos)
        extended_sinos = (extended_sinos + 9.68) * 1000  # conversion factor to uint
        extended_sinos = numpy.nan_to_num(extended_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)

        #apply phase retrieval if desired
        if self.checkBox_phase_2.isChecked() == True:
            extended_sinos = tomopy.prep.phase.retrieve_phase(extended_sinos, pixel_size=0.0001, dist=self.doubleSpinBox_distance_2.value(), energy=self.doubleSpinBox_Energy_2.value(), alpha=self.doubleSpinBox_alpha_2.value(), pad=True, ncore=None, nchunk=None)

        #reconstruct one slice
        if self.algorithm_list.currentText() == 'FBP_CUDA':
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
            slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm=tomopy.astra, options=options)
        else:
            slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm=self.algorithm_list.currentText(),
                                  filter_name=self.filter_list.currentText())

        #cut reconstructed slice to original size
        slices = slices[:,round(self.extend_FOV * self.full_size /2) : -round(self.extend_FOV * self.full_size /2) , round(self.extend_FOV * self.full_size /2) : -round(self.extend_FOV * self.full_size /2)]
        slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)
        original_reconstruction = slices[0, :, :]

        #find and display minimum and maximum values in reconstructed slice
        print(numpy.amin(original_reconstruction))
        print(numpy.amax(original_reconstruction))
        self.min.setText(str(numpy.amin(original_reconstruction)))
        self.max.setText(str(numpy.amax(original_reconstruction)))
        print('reconstructions done')

        #display reconstructed slice
        myarray = (original_reconstruction - numpy.amin(original_reconstruction)) * self.brightness.value() / (numpy.amax(original_reconstruction) - numpy.amin(original_reconstruction))
        myarray = myarray.repeat(2, axis=0).repeat(2, axis=1)
        yourQImage = qimage2ndarray.array2qimage(myarray)
        self.test_reco.setPixmap(QPixmap(yourQImage))

        #ungrey the buttons for further use of the program
        self.buttons_activate_load()
        self.buttons_activate_reco()
        self.buttons_activate_crop_volume()
        self.buttons_activate_reco_all()


    def reconstruct_all(self):
        #grey out the buttons while program is busy
        self.buttons_deactivate_all()

        QtWidgets.QApplication.processEvents()
        print('def reconstruct complete volume')

        #ask for the output path and create it
        self.path_out_reconstructed_ask = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder for reconstructions.', self.path_klick)

        #create a folder when saving reconstructed volume as tif-files
        if self.save_tiff.isChecked() == True:
            self.path_out_reconstructed_full = self.path_out_reconstructed_ask + '/'+ self.folder_name
            os.mkdir(self.path_out_reconstructed_full)
        if self.save_hdf5.isChecked() == True:
            self.path_out_reconstructed_full = self.path_out_reconstructed_ask

        self.full_size = self.A.shape[2]
        self.number_of_projections = self.A.shape[0]
        print('X-size', self.A.shape[2])
        print('Nr of projections', self.A.shape[0])
        print('Nr of slices', self.A.shape[1])

        #calculate extension of projections to the sides
        self.extend_FOV = 2* (abs(self.COR.value() - self.A.shape[2]/2))/ (1 * self.A.shape[2]) + 0.05    # extend field of view (FOV), 0.0 no extension, 0.5 half extension to both sides (for half sided 360 degree scan!!!)
        print('extend_FOV ', self.extend_FOV)

        #check if 180° or 360°-scan
        if self.number_of_projections * self.speed_W.value() >= 270:
            self.number_of_used_projections = round(360 / self.speed_W.value())
        else:
            print('smaller than 3/2 Pi')
            self.number_of_used_projections = round(180 / self.speed_W.value())
        print('number of used projections', self.number_of_used_projections)

        #create list with x-positions
        new_list = (numpy.arange(self.number_of_used_projections) * self.speed_W.value() + self.Offset_Angle.value()) * math.pi / 180
        print(new_list.shape)

        #create list with projection angles
        center_list = [self.COR.value() + round(self.extend_FOV * self.full_size)] * (self.number_of_used_projections)
        print(len(center_list))

        #save parameters in csv-file
        file_name_parameter = self.path_out_reconstructed_full + '/' + self.folder_name + '_parameter.csv'
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


        #divide volume into blocks and reconstruct them one by one in order to save RAM
        i = 0
        while (i < math.ceil(self.A.shape[1] / self.block_size)):

            print('Reconstructing block', i + 1, 'of', math.ceil(self.A.shape[1] / self.block_size))

            #extend data, take logarithm, remove NaN-values
            ###extended_sinos = self.A[0:min(self.number_of_used_projections, self.A.shape[0]), i * self.block_size: (i + 1) * self.block_size, :]
            extended_sinos = self.A[self.last_zero_proj : min(self.last_zero_proj + self.number_of_used_projections, self.A.shape[0]), i * self.block_size: (i + 1) * self.block_size, :]

            extended_sinos = tomopy.misc.morph.pad(extended_sinos, axis=2, npad=round(self.extend_FOV * self.full_size), mode='edge')
            extended_sinos = tomopy.minus_log(extended_sinos)
            extended_sinos = (extended_sinos + 9.68) * 1000  # conversion factor to uint
            extended_sinos = numpy.nan_to_num(extended_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)

            #apply phase retrieval if desired
            if self.checkBox_phase_2.isChecked() == True:
                extended_sinos = tomopy.prep.phase.retrieve_phase(extended_sinos, pixel_size=0.0001, dist=self.doubleSpinBox_distance_2.value(), energy=self.doubleSpinBox_Energy_2.value(), alpha=self.doubleSpinBox_alpha_2.value(), pad=True, ncore=None, nchunk=None)

            #reconstruct
            if self.algorithm_list.currentText() == 'FBP_CUDA':
                options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
                slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm=tomopy.astra,
                                      options=options)
            else:
                slices = tomopy.recon(extended_sinos, new_list, center=center_list,
                                      algorithm=self.algorithm_list.currentText(),
                                      filter_name=self.filter_list.currentText())

            #crop reconstructed data
            slices = slices[:, round(self.extend_FOV * self.full_size /2): -round(self.extend_FOV * self.full_size /2), round(self.extend_FOV * self.full_size /2): -round(self.extend_FOV * self.full_size /2)]
            slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)

            #16-bit integer conversion
            if self.radioButton_16bit_integer.isChecked() == True:
                ima3 = 65535 * (slices - self.int_low.value()) / (self.int_high.value() - self.int_low.value())
                ima3 = numpy.clip(ima3, 1, 65534)
                slices_save = ima3.astype(numpy.uint16)

            if self.radioButton_32bit_float.isChecked() == True:
                slices_save = slices

            print('Reconstructed Volume is', slices_save.shape)


            #save data
            if self.save_tiff.isChecked() == True:

                # write the reconstructed block to disk as TIF-file
                a = 1
                while (a < self.block_size + 1) and (a < slices_save.shape[0] + 1):
                    self.progressBar.setValue((a + (i * self.block_size)) * 100 / self.A.shape[1])
                    QtCore.QCoreApplication.processEvents()
                    time.sleep(0.02)
                    filename2 = self.path_out_reconstructed_full + self.namepart + str(
                        a + self.crop_offset + i * self.block_size).zfill(4) + '.tif'
                    print('Writing Reconstructed Slices:', filename2)
                    slice_save = slices_save[a - 1, :, :]
                    img = Image.fromarray(slice_save)
                    img.save(filename2)
                    a = a + 1

            if self.save_hdf5.isChecked() == True:
                if i == 0:
                    # create an an hdf5-file and write the first reconstructed block into it
                    with h5py.File(self.path_out_reconstructed_full + '/' + self.folder_name + '.h5', 'w')  as f:
                        f.create_dataset("Volume", data=slices_save, maxshape=(None, slices_save.shape[1], slices_save.shape[2]))
                else:
                    # write the subsequent blocks into the hdf5-file
                    self.progressBar.setValue((i * self.block_size) * 100 / self.A.shape[1])
                    QtCore.QCoreApplication.processEvents()
                    time.sleep(0.02)
                    f = h5py.File(self.path_out_reconstructed_full + '/' + self.folder_name + '.h5', 'r+')
                    vol_proxy = f['Volume']
                    print('volume_proxy.shape', vol_proxy.shape)
                    vol_proxy.resize((vol_proxy.shape[0] + slices_save.shape[0]), axis=0)
                    vol_proxy[i * self.block_size : i * self.block_size + slices_save.shape[0] ,:,:] = slices_save


            i = i + 1

        #set progress bar to 100%
        self.progressBar.setValue(100)
        QtCore.QCoreApplication.processEvents()
        time.sleep(0.02)

        #ungrey the buttons for further use of the program
        self.buttons_activate_load()
        self.buttons_activate_reco()
        self.buttons_activate_crop_volume()
        self.buttons_activate_reco_all()
        print('Done!')


#no idea why we need this, but it wouldn't work without it ;-)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    main = On_the_fly_CT_tester()
    main.show()
    sys.exit(app.exec_())

#end of code