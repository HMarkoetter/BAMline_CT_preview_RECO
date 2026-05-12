import numpy as np
import h5py
import cv2 as cv
import algotom.rec.reconstruction as rec
import algotom.prep.filtering as filt
import algotom.prep.removal as rem
import algotom.prep.correction as corr
import algotom.util.utility as util
import h5py
import pvaccess as pva
import csv
import timeit
from PIL import Image


channel_name = 'test'
import os
from superqt import QLabeledRangeSlider, QLabeledDoubleSlider, QLabeledSlider

#filepath2x = r"/raid-ssd/CT/2024/2024_03/Markoetter/W-sphere/00009_W-sphere_8___Z155_Y3500_35000eV_2x_100ms/240326_111109_00001.h5"
#filepath10x = r"/raid-ssd/CT/2024/2024_03/Markoetter/W-sphere/00011_W-sphere_8___Z155_Y3500_35000eV_10x_100ms/240326_112852_00001.h5"
#file = h5py.File(filepath2x,'r')

import sys


from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QComboBox, QGridLayout, QWidget, QLabel, QFileDialog, QCheckBox, QDoubleSpinBox, QSpinBox
from PyQt5 import QtCore, QtGui

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("My App")
        layout = QGridLayout()

        self.load_button = QPushButton("Load hdf5")
        self.usedprojslider = QLabeledRangeSlider(QtCore.Qt.Orientation.Horizontal)
        self.usedprojslider.setMinimum(0)

        self.labelproj = QLabel('Proj')
        self.usedFFslider = QLabeledRangeSlider(QtCore.Qt.Orientation.Horizontal)
        self.usedFFslider.setMinimum(0)

        self.usedFFslider.setValue((0,20))
        self.labelFF = QLabel('FF')
        self.CORslider = QLabeledDoubleSlider(QtCore.Qt.Orientation.Horizontal)
        self.CORslider.setMaximum(2000)
        self.CORslider.setValue(1247)
        self.labelCOR = QLabel('COR')
        self.sliceslider = QLabeledSlider(QtCore.Qt.Orientation.Vertical)

        #dropdown for reconstruction method
        self.reco_method_dropdown = QComboBox()
        self.reco_method_dropdown.addItem("FBP")
        self.reco_method_dropdown.addItem("Gridrec")
        self.reco_method_dropdown.addItem("Astra")
        self.reco_method_dropdown.addItem("DFI")

        self.filtering_checkbox = QCheckBox("Fresnel filter")
        self.filtering_ratio = QDoubleSpinBox()
        self.filtering_ratio.setMaximum(2000)
        self.filtering_ratio.setMinimum(0)

        self.pad = QSpinBox()
        self.pad_text = QLabel("Padding")
        self.pad.setMaximum(2000)
        self.pad.setMinimum(0)

        self.reconstruct_all_button = QPushButton("Reconstruct everything")
        self.start_slice = QSpinBox()
        self.start_slice_label = QLabel("Start slice")
        self.end_slice = QSpinBox()
        self.end_slice_label = QLabel("End slice")

        self.remove_stripes_checkbox = QCheckBox("Remove stripes")
        self.remove_stripes_snr = QDoubleSpinBox()
        self.remove_stripes_snr.setMinimum(1)
        self.remove_stripes_snr.setMaximum(10)
        self.remove_stripes_la_size = QSpinBox()
        self.remove_stripes_la_size.setMinimum(1)
        self.remove_stripes_la_size.setMaximum(100)
        self.remove_stripes_sm_size = QSpinBox()
        self.remove_stripes_sm_size.setMinimum(1)
        self.remove_stripes_sm_size.setMaximum(100)

        self.downsample_checkbox = QCheckBox("Down-sampling")
        self.downsample_slider = QLabeledRangeSlider(QtCore.Qt.Orientation.Horizontal)
        self.downsample_spinbox_high = QSpinBox()
        self.labeldown_high = QLabel('High sampling')
        self.downsample_spinbox_low = QSpinBox()
        self.labeldown_low = QLabel('Low sampling')

        # Set the central widget of the Window.
        self.button = QPushButton("Send Reco")

        layout.addWidget(self.usedprojslider,0,0)
        layout.addWidget(self.labelproj,0,1)
        layout.addWidget(self.usedFFslider,1,0)
        layout.addWidget(self.labelFF, 1, 1)
        layout.addWidget(self.CORslider,2,0)
        layout.addWidget(self.labelCOR, 2, 1)
        layout.addWidget(self.button,3,0)
        layout.addWidget(self.load_button,3,1)
        layout.addWidget(self.sliceslider,4,0,1,2)
        layout.addWidget(self.filtering_checkbox,5,0)
        layout.addWidget(self.filtering_ratio,5,1)
        layout.addWidget(self.downsample_checkbox,6,0)
        layout.addWidget(self.downsample_slider,7,0,1,2)
        layout.addWidget(self.labeldown_high, 8, 0)
        layout.addWidget(self.downsample_spinbox_high, 8, 1)
        layout.addWidget(self.labeldown_low, 8, 2)
        layout.addWidget(self.downsample_spinbox_low, 8,3)
        layout.addWidget(self.remove_stripes_checkbox,9,0)
        layout.addWidget(self.remove_stripes_snr,9,1)
        layout.addWidget(self.remove_stripes_la_size,9,2)
        layout.addWidget(self.remove_stripes_sm_size,9,3)
        layout.addWidget(self.reco_method_dropdown,10,0)
        layout.addWidget(self.pad, 11, 0)
        layout.addWidget(self.pad_text, 11, 1)
        layout.addWidget(self.reconstruct_all_button,12,0,1,2)
        layout.addWidget(self.start_slice_label,13,0)
        layout.addWidget(self.start_slice, 13, 1)
        layout.addWidget(self.end_slice_label, 13, 2)
        layout.addWidget(self.end_slice, 13, 3)

        self.filtering_ratio.hide()
        self.downsample_slider.hide()
        self.labeldown_low.hide()
        self.labeldown_high.hide()
        self.downsample_spinbox_high.hide()
        self.downsample_spinbox_low.hide()
        self.remove_stripes_snr.hide()
        self.remove_stripes_la_size.hide()
        self.remove_stripes_sm_size.hide()

        widget = QWidget()

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        pva_image_dict = {'value': ({'booleanValue': [pva.pvaccess.ScalarType.BOOLEAN], 'byteValue':
            [pva.pvaccess.ScalarType.BYTE], 'shortValue': [pva.pvaccess.ScalarType.SHORT], 'intValue':
                                         [pva.pvaccess.ScalarType.INT], 'longValue': [pva.pvaccess.ScalarType.LONG],
                                     'ubyteValue':
                                         [pva.pvaccess.ScalarType.UBYTE],
                                     'ushortValue': [pva.pvaccess.ScalarType.USHORT], 'uintValue':
                                         [pva.pvaccess.ScalarType.UINT], 'ulongValue': [pva.pvaccess.ScalarType.ULONG],
                                     'floatValue':
                                         [pva.pvaccess.ScalarType.FLOAT],
                                     'doubleValue': [pva.pvaccess.ScalarType.DOUBLE]},), 'codec':
                              {'name': pva.pvaccess.ScalarType.STRING, 'parameters': ()}, 'compressedSize':
                              pva.pvaccess.ScalarType.LONG, 'uncompressedSize': pva.pvaccess.ScalarType.LONG,
                          'dimension':
                              [{'size': pva.pvaccess.ScalarType.INT, 'offset': pva.pvaccess.ScalarType.INT, 'fullSize':
                                  pva.pvaccess.ScalarType.INT, 'binning': pva.pvaccess.ScalarType.INT, 'reverse':
                                    pva.pvaccess.ScalarType.BOOLEAN}], 'uniqueId': pva.pvaccess.ScalarType.INT,
                          'dataTimeStamp':
                              {'secondsPastEpoch': pva.pvaccess.ScalarType.LONG,
                               'nanoseconds': pva.pvaccess.ScalarType.INT,
                               'userTag': pva.pvaccess.ScalarType.INT}, 'attribute':
                              [{'name': pva.pvaccess.ScalarType.STRING, 'value': (),
                                'descriptor': pva.pvaccess.ScalarType.STRING,
                                'sourceType': pva.pvaccess.ScalarType.INT, 'source': pva.pvaccess.ScalarType.STRING}],
                          'descriptor':
                              pva.pvaccess.ScalarType.STRING,
                          'alarm': {'severity': pva.pvaccess.ScalarType.INT, 'status':
                              pva.pvaccess.ScalarType.INT, 'message': pva.pvaccess.ScalarType.STRING}, 'timeStamp':
                              {'secondsPastEpoch': pva.pvaccess.ScalarType.LONG,
                               'nanoseconds': pva.pvaccess.ScalarType.INT, 'userTag':
                                   pva.pvaccess.ScalarType.INT},
                          'display': {'limitLow': pva.pvaccess.ScalarType.DOUBLE, 'limitHigh':
                              pva.pvaccess.ScalarType.DOUBLE, 'description': pva.pvaccess.ScalarType.STRING, 'format':
                                          pva.pvaccess.ScalarType.STRING, 'units': pva.pvaccess.ScalarType.STRING}}

        self.pv_rec = pva.PvObject(pva_image_dict)
        self.pvaServer = pva.PvaServer(channel_name, self.pv_rec)
        self.pvaServer.start()
        self.new = True
        self.usedprojslider.valueChanged.connect(self.slice_changed)
        self.usedFFslider.valueChanged.connect(self.slice_changed)
        self.button.clicked.connect(self.send_reco)
        self.sliceslider.valueChanged.connect(self.slice_changed)
        self.load_button.clicked.connect(self.load_file)
        self.reconstruct_all_button.clicked.connect(self.reconstruct_all)
        self.filtering_checkbox.toggled.connect(self.filter_show)
        self.downsample_checkbox.toggled.connect(self.downsample_show)
        self.downsample_slider.valueChanged.connect(self.slice_changed)
        self.remove_stripes_checkbox.toggled.connect(self.remove_stripes_show)
        self.reco_method_dropdown.currentIndexChanged.connect(self.choose_reco_method)
        self.choose_reco_method()

    def load_file(self):
        path_klick = QFileDialog.getOpenFileName(self, 'Select hdf5-file, please.', r"/mnt/raid-ssd/CT/2024/2024_06/Markoetter/")
        if path_klick[0]:
            print(path_klick[0])
            self.path_klick = path_klick[0]
            self.htap = self.path_klick[::-1]
            self.path_in = self.path_klick[0: len(self.htap) - self.htap.find('/') - 1: 1]
            self.ni_htap = self.path_in[::-1]
            self.last_folder = self.path_in[len(self.ni_htap) - self.ni_htap.find('/') - 1::1]
            print('self.last_folder', self.last_folder)
            self.file = h5py.File(path_klick[0],'r')
            self.usedprojslider.setMaximum(self.file['entry/data/data'].shape[0])
            self.usedprojslider.setValue((80, self.file['entry/data/data'].shape[0]))
            self.usedFFslider.setMaximum(self.file['entry/data/data'].shape[0])
            self.sliceslider.setMaximum(self.file['entry/data/data'].shape[1])
            self.start_slice.setMinimum(0)
            self.start_slice.setMaximum(self.file['entry/data/data'].shape[1])
            self.end_slice.setMinimum(0)
            self.end_slice.setMaximum(self.file['entry/data/data'].shape[1])
            self.start_slice.setValue(self.file['entry/data/data'].shape[1])
            self.end_slice.setValue(self.file['entry/data/data'].shape[1])

            print('file loaded')
        else:
            print("User cancelled the dialog.")

    def filter_show(self):
        if self.filtering_checkbox.isChecked():
            self.filtering_ratio.show()
        else:
            self.filtering_ratio.hide()

    def remove_stripes_show(self):
        if self.remove_stripes_checkbox.isChecked():
            self.remove_stripes_snr.show()
            self.remove_stripes_la_size.show()
            self.remove_stripes_sm_size.show()
        else:
            self.remove_stripes_snr.hide()
            self.remove_stripes_la_size.hide()
            self.remove_stripes_sm_size.hide()

    def choose_reco_method(self):
        if self.reco_method_dropdown.currentText() == 'FBP':
            print('FBP chosen')
            self.reco_method = rec.fbp_reconstruction
        elif self.reco_method_dropdown.currentText() == 'DFI':
            print('DFI chosen')
            self.reco_method = rec.dfi_reconstruction
        elif self.reco_method_dropdown.currentText() == 'Gridrec':
            print('Gridrec chosen')
            self.reco_method = rec.gridrec_reconstruction
        elif self.reco_method_dropdown.currentText() == 'Astra':
            print('Astra chosen')
            self.reco_method = rec.astra_reconstruction

    def downsample_show(self):
        if self.downsample_checkbox.isChecked():
            self.downsample_slider.show()
            self.labeldown_low.show()
            self.labeldown_high.show()
            self.downsample_spinbox_high.show()
            self.downsample_spinbox_low.show()
        else:
            self.downsample_slider.hide()
            self.labeldown_low.hide()
            self.labeldown_high.hide()
            self.downsample_spinbox_high.hide()
            self.downsample_spinbox_low.hide()

    def slice_changed(self):
        self.new = True
        self.downsample_slider.setMinimum(self.usedprojslider.value()[0])
        self.downsample_slider.setMaximum(self.usedprojslider.value()[1])

    def create_indices_list(self):
        self.selected_indices = []

        for i in range(self.usedprojslider.value()[0], self.usedprojslider.value()[1] + 1):
            if self.downsample_slider.value()[0] <= i <= self.downsample_slider.value()[1]:
                # Inside slider values, select every xth index
                if (i - self.downsample_slider.value()[0]) % self.downsample_spinbox_high.value() == 0:
                    self.selected_indices.append(i)
            else:
                # Outside slider values, select every yth index
                if i < self.downsample_slider.value()[0]:
                    if (self.downsample_slider.value()[0] - i) % self.downsample_spinbox_low.value()  == 0:
                        self.selected_indices.append(i)
                else:
                    if (i - self.downsample_slider.value()[1]) % self.downsample_spinbox_low.value()  == 0:
                        self.selected_indices.append(i)

    def send_reco(self):
        if self.new:
            print('loading new data')
            if self.downsample_checkbox.isChecked():
                print('using downsampling')
                self.create_indices_list()
                self.proj = self.file['entry/data/data'][self.selected_indices,
                            self.sliceslider.value(), :]
                self.angles = np.radians(self.file['entry/instrument/NDAttributes/SAMPLE_W'][self.selected_indices])
                print('using ', self.angles.shape, ' angles')
            else:
                print('loading everything')
                self.proj = self.file['entry/data/data'][self.usedprojslider.value()[0]:self.usedprojslider.value()[1],
                        self.sliceslider.value(), :]
                self.angles = np.radians(self.file['entry/instrument/NDAttributes/SAMPLE_W']
                           [self.usedprojslider.value()[0]:self.usedprojslider.value()[1]])
                print('using ', self.angles.shape, ' angles')
            self.ff = np.mean(self.file['entry/data/data'][self.usedFFslider.value()[0]:self.usedFFslider.value()[1],self.sliceslider.value(),:],axis=0)
            self.norm = np.divide(self.proj-100,self.ff-100)
            self.new = False


        if self.remove_stripes_checkbox.isChecked():
            print('removing stripes')
            self.norm_final = rem.remove_all_stripe(self.norm, self.remove_stripes_snr.value(), self.remove_stripes_la_size.value(), self.remove_stripes_sm_size.value())
        else:
            if not self.filtering_checkbox.isChecked():
                self.norm_final = self.norm

        if self.filtering_checkbox.isChecked():
            print('filtering')
            self.norm_final = filt.fresnel_filter(self.norm_final, self.filtering_ratio.value(), dim=1, pad=self.pad.value())

        print('reconstructing')
        self.original_reconstruction = self.reco_method(sinogram= self.norm_final, filter_name='shepp',
                                                        pad=self.pad.value(),
                                                        center= self.CORslider.value(),angles=self.angles,
                                                              apply_log=True)
        self.pv_rec['dimension'] = [
            {'size': self.original_reconstruction.shape[1], 'fullSize': self.original_reconstruction.shape[1], 'binning': 1},
            {'size': self.original_reconstruction.shape[0], 'fullSize': self.original_reconstruction.shape[0], 'binning': 1}]
        self.pv_rec['value'] = ({'floatValue': self.original_reconstruction.flatten()},)
        print('reco done')

    def reconstruct_all(self):
        t_start = timeit.default_timer()

        self.path_out_reconstructed_ask = QFileDialog.getExistingDirectory(self,
                                                                                     'Select folder for reconstructions.',
                                                                                     self.path_klick)
        self.last_folder = self.path_in[len(self.ni_htap) - self.ni_htap.find('/') - 1::1]
        self.folder_name = self.last_folder
        htap = self.path_klick[::-1]
        self.path_in = self.path_klick[0: len(htap) - htap.find('/') - 1: 1]
        ni_htap = self.path_in[::-1]
        self.last_folder = self.path_in[len(ni_htap) - ni_htap.find('/') - 1::1]
        print('self.last_folder', self.last_folder)
        self.namepart = self.path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 1: 1]
        self.filetype = self.path_klick[len(htap) - htap.find('.') - 1: len(htap):1]

        self.base_folder = self.path_in[: len(htap) - len(self.last_folder) - len(self.namepart) - len(self.filetype)]
        print('self.base_folder', self.base_folder)
        redlof_esab = self.base_folder[::-1]
        self.sample_folder_name = self.base_folder[len(redlof_esab) - redlof_esab.find('/') - 1::1]
        self.path_out_reconstructed_full = self.path_out_reconstructed_ask + self.sample_folder_name + self.folder_name + '_reco'
        os.makedirs(self.path_out_reconstructed_full, exist_ok=True)
        print('self.path_out_reconstructed_full', self.path_out_reconstructed_full)

        file_name_parameter = self.path_out_reconstructed_full + self.folder_name + '_parameter.csv'
        print(file_name_parameter)
        with open(file_name_parameter, mode='w', newline='') as parameter_file:
            csv_writer = csv.writer(parameter_file, delimiter='\t', quotechar=' ')
            csv_writer.writerow(['Path input                    ', self.path_in, ' '])
            csv_writer.writerow(['Path output                   ', self.path_out_reconstructed_full, ' '])
            csv_writer.writerow(['Number of used projections    ', str(self.usedprojslider.value()[1]-self.usedprojslider.value()[0]), ' '])
            csv_writer.writerow(['Center of rotation            ', str(self.CORslider.value()), ' '])
            csv_writer.writerow(['Dark field value              ', str(100), ' '])
            csv_writer.writerow(['Ring handling radius          ', 'SNR : {} ,LA size: {}, SM size: {}'.format(self.remove_stripes_snr.value(),str(self.remove_stripes_la_size.value()),self.remove_stripes_sm_size.value()), ' '])
            csv_writer.writerow(['Rotation speed [°/image]      ', str(np.rad2deg(self.angles)[101]-np.rad2deg(self.angles)[100]), ' '])
            csv_writer.writerow(['Reconstruction algorithm      ', self.reco_method_dropdown.currentText(), ' '])
            csv_writer.writerow(['Padding         ', str(self.pad.value()), ' '])
            csv_writer.writerow(['Reconstruction filter         ', 'Shepp', ' '])
            csv_writer.writerow(['Software Version              ', 'Algotom', ' '])
            csv_writer.writerow(['binning                       ', '1x1x1', ' '])


        t_load = 0.0
        t_prep = 0.0
        t_rec = 0.0
        t_save = 0.0
        self.chunk_size = 100
        self.total_slice = self.end_slice.value()-self.start_slice.value()
        self.chunk = np.clip(self.chunk_size, 1, self.total_slice)
        self.last_chunk = self.total_slice - self.chunk * (self.total_slice // self.chunk)

        self.flat_fields = np.mean(self.file['entry/data/data'][self.usedFFslider.value()[0]:self.usedFFslider.value()[1],:,:],axis=0)

        for i in np.arange(self.start_slice.value(), self.start_slice.value() + self.total_slice - self.last_chunk, self.chunk):
            self.start_sino = i
            self.stop_sino = self.start_sino + self.chunk

            print('Reconstructing slices {} to {}.'.format(self.start_sino,self.stop_sino))
            # Load data, perform flat-field correction
            t0 = timeit.default_timer()
            sinograms = corr.flat_field_correction(
                proj=self.file['entry/data/data'][self.usedprojslider.value()[0]:self.usedprojslider.value()[1], self.start_sino:self.stop_sino, :],
                flat=self.flat_fields[self.start_sino:self.stop_sino, :],
                dark=self.flat_fields[self.start_sino:self.stop_sino, :],
                use_dark=False)
            t1 = timeit.default_timer()
            t_load = t_load + t1 - t0
            if self.remove_stripes_checkbox.isChecked():
                sinograms = util.parallel_process_slices(
                    sinograms,
                    rem.remove_all_stripe,
                    [
                        self.remove_stripes_snr.value(),
                        self.remove_stripes_la_size.value(),
                        self.remove_stripes_sm_size.value()
                    ],
                    axis=1,
                    prefer="threads"
                )
                # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
                #                                                 "remove_all_stripe",
                #                                                 [self.remove_stripes_snr.value(), self.remove_stripes_la_size.value(), self.remove_stripes_sm_size.value()],
                #                                                 prefer="threads")

            if self.filtering_checkbox.isChecked():
                print('filtering')
                sinograms = filt.fresnel_filter(sinograms, self.filtering_ratio.value(), dim=1,
                                                      pad=self.pad.value())

            # Perform reconstruction
            t0 = timeit.default_timer()
            recon_imgs = self.reco_method(sinogram=sinograms, filter_name='shepp',
                             pad=self.pad.value(),
                             center=self.CORslider.value(), angles=self.angles,
                             apply_log=True)
            t1 = timeit.default_timer()
            t_rec = t_rec + t1 - t0

            # Save output
            for j in range(self.start_sino, self.stop_sino):
                out_file = self.path_out_reconstructed_full + self.namepart + ("_0000" + str(j))[-5:] + ".tif"
                img16bit = 65535 * (recon_imgs[:, j - self.start_sino, :] - (-100)) / (
                            100 - (-100))
                img = Image.fromarray(img16bit)
                img.save(out_file)

            t1 = timeit.default_timer()
            t_save = t_save + t1 - t0
            t_stop = timeit.default_timer()
            print("Done slice: {0} - {1} . Time {2}".format(self.start_sino, self.stop_sino,
                                                            t_stop - t_start))

        if self.last_chunk != 0:
            self.start_sino = self.start_slice.value() + self.total_slice - self.last_chunk
            self.stop_sino = self.start_sino + self.last_chunk

            print('Reconstructing slices {} to {}.'.format(self.start_sino, self.stop_sino))
            # Load data, perform flat-field correction
            t0 = timeit.default_timer()
            sinograms = corr.flat_field_correction(
                proj=self.file['entry/data/data'][self.usedprojslider.value()[0]:self.usedprojslider.value()[1], self.start_sino:self.stop_sino, :],
                flat=self.flat_fields[self.start_sino:self.stop_sino, :],
                dark=self.flat_fields[self.start_sino:self.stop_sino, :],
                use_dark=False)
            t1 = timeit.default_timer()
            t_load = t_load + t1 - t0
            if self.remove_stripes_checkbox.isChecked():
                # sinograms = util.apply_method_to_multiple_sinograms(sinograms,
                #                                                     "remove_all_stripe",
                #                                                     [self.remove_stripes_snr.value(),
                #                                                      self.remove_stripes_la_size.value(),
                #                                                      self.remove_stripes_sm_size.value()],
                #                                                     prefer="threads")
                sinograms = util.parallel_process_slices(
                    sinograms,
                    rem.remove_all_stripe,
                    [
                        self.remove_stripes_snr.value(),
                        self.remove_stripes_la_size.value(),
                        self.remove_stripes_sm_size.value()
                    ],
                    axis=1,
                    prefer="threads"
                )
                

            if self.filtering_checkbox.isChecked():
                print('filtering')
                sinograms = filt.fresnel_filter(sinograms, self.filtering_ratio.value(), dim=1,
                                                pad=self.pad.value())

            # Perform reconstruction
            t0 = timeit.default_timer()
            recon_imgs = self.reco_method(sinogram=sinograms, filter_name='shepp',
                                          pad=self.pad.value(),
                                          center=self.CORslider.value(), angles=self.angles,
                                          apply_log=True)
            t1 = timeit.default_timer()
            t_rec = t_rec + t1 - t0

            # Save output
            for j in range(self.start_sino, self.stop_sino):
                out_file = self.path_out_reconstructed_full + self.namepart + ("_0000" + str(j))[-5:] + ".tif"
                img16bit = 65535 * (recon_imgs[:, j - self.start_sino, :] - (-100)) / (
                        100 - (-100))
                img = Image.fromarray(img16bit)
                img.save(out_file)

            t1 = timeit.default_timer()
            t_save = t_save + t1 - t0
            t_stop = timeit.default_timer()
            print("Done slice: {0} - {1} . Time {2}".format(self.start_sino, self.stop_sino,
                                                            t_stop - t_start))

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()