import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUiType
from PIL import Image
import h5py
import tomopy
import math
import time
import os
import csv
import cv2                                      #to install package with pycharm by finding "opencv-python"
from scipy.ndimage.filters import gaussian_filter, median_filter
import pvaccess as pva                          #to install package with pycharm search for "pvapy"
import algotom.prep.removal as rem
import algotom.prep.calculation as calc


# On-the-fly-CT Reco
version =  "Version 2024.06.07 d"

#Install ImageJ-PlugIn: EPICS AreaDetector NTNDA-Viewer, look for the channel specified here under channel_name, consider multiple users on servers!!!
channel_name = 'BAMline:CTReco'
#standard_path = "C:/temp/HDF5-Reading/220130_1734_604_J1_anode_half_cell_in-situ_Z30_Y5430_15000eV_1p44um_500ms/" # '/mnt/raid/CT/2022/'
standard_path = r'C:/delete/reg_data/18_230606_2044_AlTi_F_Ref_tomo___Z25_Y6500_25000eV_10x_400ms'

Ui_on_the_fly_Window, Q_on_the_fly_Window = loadUiType('on_the_fly_CT_reco_hdf_dock_widget.ui')  # connect to the GUI for the program

class On_the_fly_CT_tester(Ui_on_the_fly_Window, Q_on_the_fly_Window):


    def __init__(self):
        super(On_the_fly_CT_tester, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('On-the-fly CT Reco')

        #connect buttons to actions
        self.pushLoad.clicked.connect(self.set_path)
        self.pushReconstruct.clicked.connect(self.check_test_button)
        self.pushReconstruct_all.clicked.connect(self.reconstruct_all)
        self.slice_number.valueChanged.connect(self.check)
        self.spinBox_DF.valueChanged.connect(self.check)
        self.spinBox_back_illumination.valueChanged.connect(self.check)
        self.spinBox_ringradius.valueChanged.connect(self.check)
        self.COR.valueChanged.connect(self.check)
        self.COR_roll.valueChanged.connect(self.check)
        self.Offset_Angle.valueChanged.connect(self.check)
        self.speed_W.valueChanged.connect(self.check)
        self.algorithm_list.currentIndexChanged.connect(self.check)
        self.comboBox_180_360.currentIndexChanged.connect(self.change_scan_type)
        self.filter_list.currentIndexChanged.connect(self.check)
        self.doubleSpinBox_distance_2.valueChanged.connect(self.check)
        self.doubleSpinBox_Energy_2.valueChanged.connect(self.check)
        self.doubleSpinBox_alpha_2.valueChanged.connect(self.check)
        self.checkBox_phase_2.stateChanged.connect(self.check)
        self.radioButton_32bit_float.clicked.connect(self.check)
        self.radioButton_16bit_integer.clicked.connect(self.check)
        self.int_low.valueChanged.connect(self.check)
        self.int_high.valueChanged.connect(self.check)
        self.spinBox_top.valueChanged.connect(self.update_window_size)
        self.spinBox_bottom.valueChanged.connect(self.update_window_size)
        self.spinBox_left.valueChanged.connect(self.update_window_size)
        self.spinBox_right.valueChanged.connect(self.update_window_size)


        self.block_size = 16        #volume will be reconstructed blockwise to reduce needed RAM
        #self.extend_FOV = 0.25      #the reconstructed area will be enlarged in order to allow off axis scans
        self.crop_offset = 0        #needed for proper volume cropping
        #self.new = 1


        #### from tomostream.py, nikitinvv git, micha
        # pva type channel that contains projection and metadata
        #self.pva_structure = pva.Channel('PCO1600:Pva1:Image')

        # create pva type pv for reconstruction by copying metadata from the data pv, but replacing the sizes
        # This way the ADViewer (NDViewer) plugin can be also used for visualizing reconstructions.
        #pva_image_data = self.pva_structure.get('')
        #pva_image_dict = pva_image_data.getStructureDict()
        pva_image_dict = {'value': ({'booleanValue': [pva.pvaccess.ScalarType.BOOLEAN], 'byteValue':
            [pva.pvaccess.ScalarType.BYTE], 'shortValue': [pva.pvaccess.ScalarType.SHORT], 'intValue':
            [pva.pvaccess.ScalarType.INT], 'longValue': [pva.pvaccess.ScalarType.LONG], 'ubyteValue':
            [pva.pvaccess.ScalarType.UBYTE], 'ushortValue': [pva.pvaccess.ScalarType.USHORT], 'uintValue':
            [pva.pvaccess.ScalarType.UINT], 'ulongValue': [pva.pvaccess.ScalarType.ULONG], 'floatValue':
            [pva.pvaccess.ScalarType.FLOAT], 'doubleValue': [pva.pvaccess.ScalarType.DOUBLE]},), 'codec':
            {'name': pva.pvaccess.ScalarType.STRING, 'parameters': ()}, 'compressedSize':
            pva.pvaccess.ScalarType.LONG, 'uncompressedSize': pva.pvaccess.ScalarType.LONG, 'dimension':
            [{'size': pva.pvaccess.ScalarType.INT, 'offset': pva.pvaccess.ScalarType.INT, 'fullSize':
                pva.pvaccess.ScalarType.INT, 'binning': pva.pvaccess.ScalarType.INT, 'reverse':
                pva.pvaccess.ScalarType.BOOLEAN}], 'uniqueId': pva.pvaccess.ScalarType.INT, 'dataTimeStamp':
            {'secondsPastEpoch': pva.pvaccess.ScalarType.LONG, 'nanoseconds': pva.pvaccess.ScalarType.INT,
             'userTag': pva.pvaccess.ScalarType.INT}, 'attribute':
            [{'name': pva.pvaccess.ScalarType.STRING, 'value': (), 'descriptor': pva.pvaccess.ScalarType.STRING,
              'sourceType': pva.pvaccess.ScalarType.INT, 'source': pva.pvaccess.ScalarType.STRING}], 'descriptor':
            pva.pvaccess.ScalarType.STRING, 'alarm': {'severity': pva.pvaccess.ScalarType.INT, 'status':
            pva.pvaccess.ScalarType.INT, 'message': pva.pvaccess.ScalarType.STRING}, 'timeStamp':
            {'secondsPastEpoch': pva.pvaccess.ScalarType.LONG, 'nanoseconds': pva.pvaccess.ScalarType.INT, 'userTag':
                pva.pvaccess.ScalarType.INT}, 'display': {'limitLow': pva.pvaccess.ScalarType.DOUBLE, 'limitHigh':
            pva.pvaccess.ScalarType.DOUBLE, 'description': pva.pvaccess.ScalarType.STRING, 'format':
            pva.pvaccess.ScalarType.STRING, 'units': pva.pvaccess.ScalarType.STRING}}

        self.pv_rec = pva.PvObject(pva_image_dict)
        self.pvaServer = pva.PvaServer(channel_name, self.pv_rec)
        self.Qchannel_name.setText(channel_name)
        self.pvaServer.start()



    def check(self):    #AUTO UPDATE ON/OFF?

         if self.auto_update.isChecked():
             self.check_test_button()
         return

    def update_window_size(self):
        self.new = 1
        #update possible range for crop inputs
        print(self.slice_size)
        self.spinBox_left.setMaximum(self.slice_size-self.spinBox_right.value() -1)
        self.spinBox_right.setMaximum(self.slice_size-self.spinBox_left.value() -1)
        self.spinBox_top.setMaximum(self.slice_size-self.spinBox_bottom.value() -1)
        self.spinBox_bottom.setMaximum(self.slice_size-self.spinBox_top.value() -1)
        self.check()

    def check_test_button(self):
        #check what is still in RAM and does not need to be updated
         if self.slice_in_ram != self.slice_number.value() or self.ringradius_in_RAM != self.spinBox_ringradius.value() or self.spinBox_DF_in_ram != self.spinBox_DF.value() or self.spinBox_back_illumination_in_ram != self.spinBox_back_illumination.value():
             self.load()
         else:
             self.reconstruct()
         return



    def buttons_deactivate_all(self):
        self.comboBox_180_360.setEnabled(False)
        self.spinBox_ringradius.setEnabled(False)
        self.spinBox_DF.setEnabled(False)
        self.spinBox_back_illumination.setEnabled(False)
        self.pushLoad.setEnabled(False)

        self.slice_number.setEnabled(False)
        self.COR.setEnabled(False)
        self.COR_roll.setEnabled(False)
        self.Offset_Angle.setEnabled(False)
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
        self.spinBox_left.setEnabled(False)
        self.spinBox_right.setEnabled(False)
        self.spinBox_top.setEnabled(False)
        self.spinBox_bottom.setEnabled(False)
        #self.push_Crop_volume.setEnabled(False)

        self.pushReconstruct_all.setEnabled(False)
        self.int_low.setEnabled(False)
        self.int_high.setEnabled(False)
        self.hdf_chunking_x.setEnabled(False)
        self.hdf_chunking_y.setEnabled(False)

    def buttons_activate_load(self):
        self.comboBox_180_360.setEnabled(True)
        self.spinBox_ringradius.setEnabled(True)
        self.spinBox_DF.setEnabled(True)
        self.spinBox_back_illumination.setEnabled(True)
        self.pushLoad.setEnabled(True)

    def buttons_activate_reco(self):
        self.slice_number.setEnabled(True)
        self.COR.setEnabled(True)
        self.COR_roll.setEnabled(True)
        self.Offset_Angle.setEnabled(True)
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
        self.hdf_chunking_x.setEnabled(True)
        self.hdf_chunking_y.setEnabled(True)
        self.radioButton_16bit_integer.setEnabled(True)
        self.radioButton_32bit_float.setEnabled(True)
        self.save_tiff.setEnabled(True)
        self.save_hdf5.setEnabled(True)
        self.auto_update.setEnabled(True)
        self.spinBox_first.setEnabled(True)
        self.spinBox_last.setEnabled(True)

    def buttons_activate_crop_volume(self):
        self.spinBox_first.setEnabled(True)
        self.spinBox_last.setEnabled(True)
        self.spinBox_left.setEnabled(True)
        self.spinBox_right.setEnabled(True)
        self.spinBox_top.setEnabled(True)
        self.spinBox_bottom.setEnabled(True)
        #self.push_Crop_volume.setEnabled(True)



    def set_path(self):
        #grey out the buttons while program is busy
        self.buttons_deactivate_all()
        self.pushReconstruct.setText('Busy')
        self.pushReconstruct_all.setText('Busy\n')
        self.new = 1
        self.extend_FOV_fixed_ImageJ_Stream = 0.05

        #ask for hdf5-file
        path_klick = QtWidgets.QFileDialog.getOpenFileName(self, 'Select hdf5-file, please.', standard_path)

        if path_klick[0]:
            self.path_klick = path_klick[0]
            print('path klicked: ', self.path_klick)
            self.cut_path_name()
        else:
            print("User cancelled the dialog.")
            self.buttons_activate_load()



    def cut_path_name(self):
        #analyse and cut the path in pieces and get relevant information from raw-file
        htap = self.path_klick[::-1]
        self.path_in = self.path_klick[0: len(htap) - htap.find('/') - 1: 1]
        ni_htap = self.path_in[::-1]
        self.last_folder = self.path_in[len(ni_htap) - ni_htap.find('/') - 1 :  :1]
        print('self.last_folder', self.last_folder)

        self.namepart = self.path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 1: 1]
        self.filetype = self.path_klick[len(htap) - htap.find('.') - 1: len(htap):1]


        self.base_folder = self.path_in[ : len(htap) - len(self.last_folder) - len(self.namepart) - len(self.filetype)]
        print('self.base_folder', self.base_folder)
        redlof_esab = self.base_folder[::-1]

        self.sample_folder_name = self.base_folder[len(redlof_esab) - redlof_esab.find('/') - 1 :  :1]
        print('sample_folder_name', self.sample_folder_name)

        print('chopped path: ',self.path_in, '  ', self.last_folder,'  ', self.namepart,'  ', self.filetype)
        self.Sample.setText(self.path_klick)

        #link a volume to the hdf-file
        f = h5py.File(self.path_klick, 'r')
        self.vol_proxy = f['/entry/data/data']
        print('raw data volume size: ', self.vol_proxy.shape)

        #get the image dimensions and prefill slice number
        print('self.vol_proxy.shape[1] ', self.vol_proxy.shape[1])
        print('round(self.vol_proxy.shape[1]/2) ', round(self.vol_proxy.shape[1]/2))

        self.slice_number.setMaximum(self.vol_proxy.shape[1]-1)
        print('set Maximum')
        self.slice_number.setMinimum(0)
        print('set Minimum')
        time.sleep(1)
        self.slice_number.setValue(round(self.vol_proxy.shape[1]/2))    # be careful with an infinite loop when setValue actually triggers valueChanged. Therefore, auto_update starts off
        print('set middle height as slice number:  ', self.slice_number.value())
        self.slice_number.setEnabled(True)

        #get rotation angles
        #self.line_proxy = f['/entry/instrument/NDAttributes/CT_MICOS_W']
        self.line_proxy = f['/entry/instrument/NDAttributes/SAMPLE_MICOS_W2']
        print('self.line_proxy', self.line_proxy)
        if self.FF_before_after_checkbox.isChecked():
            print('FF before after')
            self.graph = numpy.array(self.line_proxy[self.spinBox_number_FFs.value():-self.spinBox_number_FFs.value()])
        else:
            self.graph = numpy.array(self.line_proxy[self.spinBox_number_FFs.value():])
        print('found number of angles:  ', self.graph.shape, '      angles: ', self.graph)

        #find rotation start
        i = 0
        while round(self.graph[i]) < 1:  # notice the last projection at below 0.5°
            print('angle value :',self.graph[i], 'rounded value',round(self.graph[i]))
            self.last_zero_proj = i + 3  # assumes 3 images for speeding up the motor
            i = i + 1
        #print(self.graph[1021:1500])
        print('Last projection at 0 degree/still speeding up: number', self.last_zero_proj)

        if self.COR.value() == 0:
            self.COR.setValue(round(self.vol_proxy.shape[2] / 2))
            print('COR.setValue', round(self.vol_proxy.shape[2] / 2))

        #prefill cropping
        self.spinBox_first.setValue(0)
        self.spinBox_last.setValue(self.vol_proxy.shape[1]-1)
        print('set possible crop range')

        #get and prefill pixel_size
        if '/entry/instrument/NDAttributes/CT_Pixelsize' in f:
            self.pixel_proxy = f['/entry/instrument/NDAttributes/CT_Pixelsize']
            print('self.pixel_proxy', self.pixel_proxy[0], self.pixel_proxy[-1])
            self.pixel_size.setValue(self.pixel_proxy[0])
        else:
            self.pixel_size.setValue(1)
            print('pixel size not found')
        self.pixel_size.setEnabled(True)

        #get and prefill energy
        if '/entry/instrument/NDAttributes/DMM_Energy' in f:
            self.energy_proxy = f['/entry/instrument/NDAttributes/DMM_Energy']
            print('self.energy_proxy', self.energy_proxy[0], self.energy_proxy[-1])
            self.doubleSpinBox_Energy_2.setValue(self.energy_proxy[0])
        else:
            self.doubleSpinBox_Energy_2.setValue(1)
            print('Energy size not found')
        self.doubleSpinBox_Energy_2.setEnabled(True)

        #get and prefill distance
        if '/entry/instrument/NDAttributes/CT-Kamera-Z' in f:
            self.distance_proxy = f['/entry/instrument/NDAttributes/CT-Kamera-Z']
            print('self.distance_proxy', self.distance_proxy[0], self.distance_proxy[-1])
            self.doubleSpinBox_distance_2.setValue((self.distance_proxy[0] + 25)/10)
        else:
            self.doubleSpinBox_distance_2.setValue(1)
            print('Distance not found')
        self.doubleSpinBox_distance_2.setEnabled(True)

        self.load()


    def change_scan_type(self):
        self.new = 1

        if self.comboBox_180_360.currentText() == '180 - axis centered':
            self.extend_FOV_fixed_ImageJ_Stream = 0.05
        else:
            self.extend_FOV_fixed_ImageJ_Stream = 1.15
        print('extend FOV',self.extend_FOV_fixed_ImageJ_Stream)

        self.check()


    def load(self):
        self.buttons_deactivate_all()

        FFs = self.vol_proxy[0:self.spinBox_number_FFs.value() -1, self.slice_number.value(), :]
        FFmean = numpy.mean(FFs, axis=0)
        print('FFs for normalization ', self.spinBox_number_FFs.value(), FFmean.shape)
        if self.FF_before_after_checkbox.isChecked():
            Sino = self.vol_proxy[self.spinBox_number_FFs.value():-self.spinBox_number_FFs.value(), self.slice_number.value(), :]
        else:
            Sino = self.vol_proxy[self.spinBox_number_FFs.value() :, self.slice_number.value(), :]
        self.Norm = numpy.divide(numpy.subtract(Sino, self.spinBox_DF.value()), numpy.subtract(FFmean, self.spinBox_DF.value() + self.spinBox_back_illumination.value()))
        #self.Norm = numpy.divide(Sino, FFmean)
        print('Norm shape', self.Norm.shape)
        self.spinBox_DF_in_ram = self.spinBox_DF.value()
        self.spinBox_back_illumination_in_ram = self.spinBox_back_illumination.value()
        self.slice_in_ram = self.slice_number.value()
        self.ringradius_in_RAM = self.spinBox_ringradius.value()
        print('set Slice in RAM')
        self.w = self.graph     #no need to load the angles each time a new slice is picked
        print('set angles in self.w')

        #Ring artifact handling

        if self.spinBox_ringradius.value() != 0:
            print('ring handling')

            #self.Norm = rem.remove_stripe_based_sorting(self.Norm, size=self.spinBox_ringradius.value())
            #self.Norm = rem.remove_all_stripe(self.Norm, snr=3.1, la_size=self.spinBox_ringradius.value(), sm_size=21) # ONLY 2D????


            self.proj_sum = numpy.mean(self.Norm, axis = 0)
            self.proj_sum_2d = numpy.zeros((1, self.proj_sum.shape[0]), dtype = numpy.float32)
            self.proj_sum_2d[0,:] = self.proj_sum
            print('proj_sum dimensions', self.proj_sum.shape)
            print('proj_sum_2d dimensions', self.proj_sum_2d.shape)

            proj_sum_filtered = median_filter(self.proj_sum_2d, size = (1,self.spinBox_ringradius.value()), mode='nearest')
            print('proj_sum_filtered dimensions', proj_sum_filtered.shape)
            correction_map = numpy.divide(self.proj_sum_2d, proj_sum_filtered)
            correction_map = numpy.clip(correction_map, 0.5, 2.0)
            print('correction_map dimensions', correction_map.shape, 'correction_map min vs max', numpy.amin(correction_map), numpy.amax(correction_map))

            i=0
            while i < self.Norm.shape[0]:
                self.Norm[i, :] = numpy.divide(self.Norm[i, :], correction_map[0,:])
                self.progressBar.setValue((i + 1) * 100 / self.Norm.shape[0])
                QtWidgets.QApplication.processEvents()
                i = i+1



            print('Norm.shape', self.Norm.shape)
            print('finished ring handling')
        else:
            print('did not do ring handling')





        #prefill rotation-speed[°/img]
        #Polynom fit for the angles, changed /4 to /2 and 3/4 to 7/8
        poly_coeff = numpy.polyfit(numpy.arange(len(self.w[round((self.w.shape[0] + 1) /2) : round((self.w.shape[0] + 1) * 7/8) ])), self.w[round((self.w.shape[0] + 1) /2) : round((self.w.shape[0] + 1) * 7/8) ], 1, rcond=None, full=False, w=None, cov=False)
        print('Polynom coefficients',poly_coeff, '   Detected angular step per image: ', poly_coeff[0])
        self.speed_W.setValue(poly_coeff[0])
        print('Last projection at 0 degree/still speeding up: image number', self.last_zero_proj)

        time.sleep(1) #???

        #ungrey the buttons for further use of the program
        self.buttons_activate_load()
        self.buttons_activate_reco()
        self.buttons_activate_crop_volume()
        self.buttons_activate_reco_all()
        print('Loading/changing slice complete!')

        self.reconstruct()



    def reconstruct(self):

        # grey out the buttons while program is busy
        self.buttons_deactivate_all()
        self.pushReconstruct.setText('Busy')
        self.pushReconstruct_all.setText('Busy\n')


        QtWidgets.QApplication.processEvents()
        #print('def reconstruct')

        self.full_size = self.Norm.shape[1]
        self.number_of_projections = self.Norm.shape[0]

        #check if the scan was 180° or 360°
        if self.graph[-1] >= 350:
            self.number_of_used_projections = round(360 / self.speed_W.value())
            print('360°')
        else:
            #print('smaller than 3/2 Pi')
            self.number_of_used_projections = round(180 / self.speed_W.value())
            print('180°')
        print('number of projections used for reconstruction (omitting those above 180°/360°: )', self.number_of_used_projections)

        # create list with all projection angles
        new_list = (numpy.arange(self.number_of_used_projections) * self.speed_W.value() + self.Offset_Angle.value()) * math.pi / 180

        # create list with x-positions of projections
        if self.comboBox_180_360.currentText() == '360 - axis right':
            center_list = [self.COR.value() + self.COR_roll.value() * self.slice_number.value() + round((self.extend_FOV_fixed_ImageJ_Stream -1) * self.full_size)] * (self.number_of_used_projections)
            #center_list = [self.COR.value() +  self.full_size] * (self.number_of_used_projections)
        else:
            center_list = [self.COR.value() + self.COR_roll.value() * self.slice_number.value() + round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size)] * (self.number_of_used_projections)

        # create one sinogram in the form [z, y, x]
        transposed_sinos = numpy.zeros((min(self.number_of_used_projections, self.Norm.shape[0]), 1, self.full_size), dtype=float)
        transposed_sinos[:,0,:] = self.Norm[self.last_zero_proj : min(self.last_zero_proj + self.number_of_used_projections, self.Norm.shape[0]),:]



        #extend data with calculated parameter, compute logarithm, remove NaN-values
        ### cut and reorder 360°-sinos to 180°-sinos, work in progress !!!
        extended_sinos = tomopy.misc.morph.pad(transposed_sinos, axis=2, npad=round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size), mode='edge')

        # for 360° scans crop the padded area opposite of the axis
        print('cropping empty area')
        print(extended_sinos.shape)
        if self.comboBox_180_360.currentText() == '360 - axis right':
            #extended_sinos = extended_sinos[:,:, round((self.extend_FOV_fixed_ImageJ_Stream -1) * self.full_size) : ]
            extended_sinos = extended_sinos[:,:, self.full_size : ]

        elif self.comboBox_180_360.currentText() == '360 - axis left':
            #extended_sinos = extended_sinos[:,:, : - round((self.extend_FOV_fixed_ImageJ_Stream -1) * self.full_size)]
            extended_sinos = extended_sinos[:,:, : - self.full_size]

        print(extended_sinos.shape)

        extended_sinos = tomopy.minus_log(extended_sinos)

        extended_sinos = numpy.nan_to_num(extended_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
        print('extended_sinos.shape', extended_sinos.shape)

        #apply phase retrieval if desired
        if self.checkBox_phase_2.isChecked() == True:
            print('applying phase contrast')
            extended_sinos = tomopy.prep.phase.retrieve_phase(extended_sinos, pixel_size=0.0001, dist=self.doubleSpinBox_distance_2.value(), energy=self.doubleSpinBox_Energy_2.value(), alpha=self.doubleSpinBox_alpha_2.value(), pad=True, ncore=None, nchunk=None)

        print('ring_filter sino_shape', extended_sinos[:,0,:].shape)

        #extended_sinos[:,0,:] = rem.remove_stripe_based_sorting(extended_sinos[:,0,:], size=21)
        #extended_sinos = algotom.prep.removal.remove_stripe_based_filtering(extended_sinos, sigma=3, size=21)
        #center = calc.find_center_vo(extended_sinos[:,0,:], extended_sinos.shape[2] // 2 - 50, extended_sinos.shape[2] // 2 + 50)
        #print('AUTO-CENTER:', center + round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size))


        #reconstruct one slice
        if self.algorithm_list.currentText() == 'FBP_CUDA':
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
            slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm=tomopy.astra, options=options)
        else:
            slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm=self.algorithm_list.currentText(),
                                  filter_name=self.filter_list.currentText())

        # scale with pixel size to attenuation coefficients
        slices = slices * (10000/self.pixel_size.value())
        print('slices.shape', slices.shape)

        # trim reconstructed slice
        if self.comboBox_180_360.currentText() == '180 - axis centered':
            slices = slices[:, round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size / 2): -round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size / 2),round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size / 2): -round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size / 2)]
        else:
            slices = slices[:, round((self.extend_FOV_fixed_ImageJ_Stream -1) * self.full_size): -round((self.extend_FOV_fixed_ImageJ_Stream -1) * self.full_size),round((self.extend_FOV_fixed_ImageJ_Stream -1) * self.full_size): -round((self.extend_FOV_fixed_ImageJ_Stream -1) * self.full_size)]
        self.slice_size = slices.shape[1]
        slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)

        self.slice = slices[0,:,:]   #reduce dimensions from 3 to 2

        if self.checkBox_ruler.isChecked():
            self.add_ruler()


        #crop before sending to ImageJ      spinBox_left
        original_reconstruction = self.slice[self.spinBox_top.value():-self.spinBox_bottom.value()-1, self.spinBox_left.value():-self.spinBox_right.value()-1]
        print('original_reconstruction.shape', original_reconstruction.shape)

        # set image dimensions only for the first time or when scan-type was changed
        if self.new == 1:
            self.pv_rec['dimension'] = [
                {'size': original_reconstruction.shape[1], 'fullSize': original_reconstruction.shape[1], 'binning': 1},
                {'size': original_reconstruction.shape[0], 'fullSize': original_reconstruction.shape[0], 'binning': 1}]
            self.new = 0

        # 16-bit integer conversion
        if self.radioButton_16bit_integer.isChecked() == True:
            self.ima3 = 65535 * (original_reconstruction - self.int_low.value()) / (self.int_high.value() - self.int_low.value())
            self.ima3 = numpy.clip(self.ima3, 1, 65534)
            self.ima3 = numpy.around(self.ima3)
            print('slice_show.shape',self.ima3.shape)

            # write result to pv
            self.pv_rec['value'] = ({'floatValue': self.ima3.flatten()},)
            print('Reconstructed Slice is', self.ima3.shape)

        # keep 32-bit float
        if self.radioButton_32bit_float.isChecked() == True:
            self.slice_show = original_reconstruction

            # write result to pv
            self.pv_rec['value'] = ({'floatValue': self.slice_show.flatten()},)
            print('Reconstructed Slice is', self.slice_show.shape)


        #find and display minimum and maximum values in reconstructed slice
        print('minimum value found: ', numpy.amin(original_reconstruction), '     maximum value found: ',numpy.amax(original_reconstruction))

        print('reconstruction of slice is done')

        #ungrey the buttons for further use of the program
        self.buttons_activate_load()
        self.buttons_activate_reco()
        self.buttons_activate_crop_volume()
        self.buttons_activate_reco_all()
        self.pushReconstruct.setText('Test')
        self.pushReconstruct_all.setText('Reconstruct\n Volume')

    def add_ruler(self):
        f = h5py.File(self.path_klick, 'r')
        if '/entry/instrument/NDAttributes/CT_Piezo_X45' in f:
            self.piezo_45_proxy = f['/entry/instrument/NDAttributes/CT_Piezo_X45']
            print('self.piezo_45_proxy: ', self.piezo_45_proxy[0], self.piezo_45_proxy[-1],
                  round(1000 * self.piezo_45_proxy[-1]))
            self.piezo_135_proxy = f['/entry/instrument/NDAttributes/CT_Piezo_Y45']
            print('self.piezo_135_proxy: ', self.piezo_135_proxy[0], self.piezo_135_proxy[-1],
                  round(1000 * self.piezo_135_proxy[-1]))
        else:
            self.piezo_45_proxy = (0,0)
            self.piezo_135_proxy = (0,0)

        # draws a circle with the detector size as diameter
        cv2.circle(self.slice, (round(self.slice.shape[1] / 2), round(self.slice.shape[1] / 2)), 1280, (255, 255, 255), 4)

        if self.grid_micrometer.isChecked() == True:
            print(self.pixel_size.value())
            # add ruler +X
            for r in range(round(self.slice.shape[1] * self.pixel_size.value() / 2),
                           round(self.slice.shape[1] * self.pixel_size.value()),
                           round(self.spinBox_ruler_grid.value())):
                cv2.line(self.slice, (round(r / self.pixel_size.value()), 0),
                         (round(r / self.pixel_size.value()), self.slice.shape[0]), (65535, 65535, 65535), 4)
                cv2.putText(self.slice, str(round(1000 * self.piezo_45_proxy[-1]/  5) * 5   +   round( r / 5) * 5   -   round(self.slice.shape[1] * (self.pixel_size.value() / 10)) * 5),
                            (round(r / self.pixel_size.value()) + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.750,
                            (65535, 65535, 65535), thickness=2)

            # add ruler -X
            for r in range(round(self.slice.shape[1] * self.pixel_size.value() / 2), 0,
                           -round(self.spinBox_ruler_grid.value())):
                cv2.line(self.slice, (round(r / self.pixel_size.value()), 0),
                         (round(r / self.pixel_size.value()), self.slice.shape[0]), (65535, 65535, 65535), 4)
                cv2.putText(self.slice, str(round(1000 * self.piezo_45_proxy[-1] / 5) * 5  +  round(
                    r  / 5) * 5  -  round(self.slice.shape[1] * (self.pixel_size.value() / 10)) * 5),
                            (round(r / self.pixel_size.value()) + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.750,
                            (65535, 65535, 65535), thickness=2)

            # add ruler +Y
            for r in range(round(self.slice.shape[1] * self.pixel_size.value() / 2),
                           round(self.slice.shape[1] * self.pixel_size.value()),
                           round(self.spinBox_ruler_grid.value())):
                cv2.line(self.slice, (0, round(r / self.pixel_size.value())),
                         (self.slice.shape[1], round(r / self.pixel_size.value())), (65535, 65535, 65535), 4)
                cv2.putText(self.slice, str(round(1000 * self.piezo_135_proxy[-1] / 5) * 5  +  round(
                    r / 5) * 5 - round(self.slice.shape[1] * (self.pixel_size.value() / 10)) * 5),
                            (20, round(r / self.pixel_size.value()) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.750,
                            (65535, 65535, 65535), thickness=2)

            # add ruler -Y
            for r in range(round(self.slice.shape[1] * self.pixel_size.value() / 2), 0,
                           -round(self.spinBox_ruler_grid.value())):
                cv2.line(self.slice, (0, round(r / self.pixel_size.value())),
                         (self.slice.shape[1], round(r / self.pixel_size.value())), (65535, 65535, 65535), 4)
                cv2.putText(self.slice, str(round(1000 * self.piezo_135_proxy[-1] / 5) * 5 + round(
                    r / 5) * 5 - round(self.slice.shape[1] * (self.pixel_size.value()/ 10)) * 5),
                            (20, round(r / self.pixel_size.value()) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.750,
                            (65535, 65535, 65535), thickness=2)

        if self.grid_pixel.isChecked() == True:

            # add ruler +X
            for r in range(round(self.slice.shape[1] / 2), self.slice.shape[1], round(self.spinBox_pixel_grid.value())):
                cv2.line(self.slice, (r, 0),   (r, self.slice.shape[1]), (65535, 65535, 65535), 4)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (r + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.750, (65535, 65535, 65535), thickness=2)

            # add ruler -X
            for r in range(round(self.slice.shape[1] / 2), 0, -round(self.spinBox_pixel_grid.value())):
                print(r)
                cv2.line(self.slice, (r, 0),   (r, self.slice.shape[1]), (65535, 65535, 65535), 4)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (r + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.750, (65535, 65535, 65535), thickness=2)

            # add ruler +Y
            for r in range(round(self.slice.shape[1] / 2), self.slice.shape[1], round(self.spinBox_pixel_grid.value())):
                cv2.line(self.slice, (0, r), (self.slice.shape[1], r), (65535, 65535, 65535), 4)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (20,r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.750, (65535, 65535, 65535), thickness=2)

            # add ruler -Y
            for r in range(round(self.slice.shape[1] / 2), 0, -round(self.spinBox_pixel_grid.value())):
                cv2.line(self.slice, (0, r), (self.slice.shape[1], r), (65535, 65535, 65535), 4)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (20,r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.750, (65535, 65535, 65535), thickness=2)


    def reconstruct_all(self):
        #grey out the buttons while program is busy
        self.buttons_deactivate_all()
        self.pushReconstruct.setText('Busy')
        self.pushReconstruct_all.setText('Busy\n')

        self.progressBar.setValue(0)
        QtCore.QCoreApplication.processEvents()

        QtWidgets.QApplication.processEvents()
        print('def reconstruct complete volume')

        #ask for the output path and create it
        self.path_out_reconstructed_ask = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select folder for reconstructions.', self.path_klick)

        self.folder_name = self.last_folder

        #create a folder when saving reconstructed volume as tif-files
        if self.save_tiff.isChecked() == True:
            self.path_out_reconstructed_full = self.path_out_reconstructed_ask + self.sample_folder_name + self.folder_name +'_reco'
        if self.save_hdf5.isChecked() == True:
            self.path_out_reconstructed_full = self.path_out_reconstructed_ask + self.sample_folder_name

        os.makedirs(self.path_out_reconstructed_full, exist_ok = True)
        print('self.path_out_reconstructed_full', self.path_out_reconstructed_full)

        #determine how far to extend field of view (FOV), 0.0 no extension, 0.5 half extension, 1.0 full extension to both sides (for off center 360 degree scans!!!)
        self.extend_FOV = (2 * (abs(self.COR.value() - self.Norm.shape[1]/2))/ (self.Norm.shape[1])) + 0.15
        print('extend_FOV ', self.extend_FOV)

        #check if 180° or 360°-scan
        if self.number_of_projections * self.speed_W.value() >= 270:
            self.number_of_used_projections = round(360 / self.speed_W.value())
        else:
            print('smaller than 3/2 Pi')
            self.number_of_used_projections = round(180 / self.speed_W.value())
        print('number of used projections', self.number_of_used_projections)

        # create list with projection angles
        new_list = (numpy.arange(self.number_of_used_projections) * self.speed_W.value() + self.Offset_Angle.value()) * math.pi / 180
        print(new_list.shape)



        #save parameters in csv-file
        file_name_parameter = self.path_out_reconstructed_full + self.folder_name + '_parameter.csv'
        print(file_name_parameter)
        with open(file_name_parameter, mode = 'w', newline='') as parameter_file:
            csv_writer = csv.writer(parameter_file, delimiter = '\t', quotechar=' ')
            csv_writer.writerow(['Path input                    ', self.path_in,' '])
            csv_writer.writerow(['Path output                   ', self.path_out_reconstructed_full,' '])
            csv_writer.writerow(['Number of used projections    ', str(self.number_of_used_projections),' '])
            csv_writer.writerow(['Center of rotation            ', str(self.COR.value()), ' '])
            csv_writer.writerow(['Center of rotation_roll       ', str(self.COR_roll.value()), ' '])
            csv_writer.writerow(['Dark field value              ', str(self.spinBox_DF.value()),' '])
            csv_writer.writerow(['Back illumination value       ', str(self.spinBox_back_illumination.value()),' '])
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
        #self.vol_proxy_crop = self.vol_proxy[:,self.spinBox_first.value():self.spinBox_last.value(),:]
        i = 0
        while (i < math.ceil((self.spinBox_last.value() - self.spinBox_first.value()) / self.block_size)):       #Need to fix the rest of slices

            print('Reconstructing block', i + 1, 'of', math.ceil((self.spinBox_last.value() - self.spinBox_first.value()) / self.block_size))

            FFs_vol = self.vol_proxy[0:self.spinBox_number_FFs.value() - 1, i * self.block_size + self.spinBox_first.value(): (i + 1) * self.block_size + self.spinBox_first.value(), :]
            FFmean_vol = numpy.mean(FFs_vol, axis=0)
            print('FFs for normalization ', self.spinBox_number_FFs.value(), FFmean_vol.shape)
            Sino_vol = self.vol_proxy[self.spinBox_number_FFs.value(): -self.spinBox_number_FFs.value(), i * self.block_size + self.spinBox_first.value(): (i + 1) * self.block_size + self.spinBox_first.value(), :]
            self.Norm_vol = numpy.divide(Sino_vol - self.spinBox_DF.value(), FFmean_vol - self.spinBox_DF.value() - self.spinBox_back_illumination.value())
            print('sinogram shape', self.Norm_vol.shape)


            # Ring artifact handling
            if self.spinBox_ringradius.value() != 0:

                self.proj_sum = numpy.mean(self.Norm_vol, axis=0)
                print('proj_sum dimensions', self.proj_sum.shape)
                proj_sum_filtered = median_filter(self.proj_sum, size= (1, self.spinBox_ringradius.value()), mode='nearest')
                print('proj_sum_filtered dimensions', proj_sum_filtered.shape)
                correction_map_vol = numpy.divide(self.proj_sum, proj_sum_filtered)
                correction_map_vol = numpy.clip(correction_map_vol, 0.5, 2.0)
                print('correction_map_vol dimensions', correction_map_vol.shape, 'correction_map min vs max',
                      numpy.amin(correction_map_vol), numpy.amax(correction_map_vol))

                j = 0
                while j < self.Norm_vol.shape[0]:
                    self.Norm_vol[j, :,:] = numpy.divide(self.Norm_vol[j, :,:], correction_map_vol)
                    j = j + 1
                print('Norm_vol.shape', self.Norm_vol.shape)


                print('finished ring handling')
            else:
                print('did not do ring handling')




            #extend data, take logarithm, remove NaN-values
            extended_sinos = self.Norm_vol[self.last_zero_proj : min(self.last_zero_proj + self.number_of_used_projections, self.Norm_vol.shape[0]), :, :]

            extended_sinos = tomopy.misc.morph.pad(extended_sinos, axis=2, npad=round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size), mode='edge')
            extended_sinos = tomopy.minus_log(extended_sinos)
            #extended_sinos = (extended_sinos + 9.68) * 1000  # conversion factor to uint
            extended_sinos = numpy.nan_to_num(extended_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)

            #apply phase retrieval if desired
            if self.checkBox_phase_2.isChecked() == True:
                extended_sinos = tomopy.prep.phase.retrieve_phase(extended_sinos, pixel_size=0.0001, dist=self.doubleSpinBox_distance_2.value(), energy=self.doubleSpinBox_Energy_2.value(), alpha=self.doubleSpinBox_alpha_2.value(), pad=True, ncore=None, nchunk=None)

            # create list with COR-positions
            center_list = [self.COR.value() + self.COR_roll.value() * (i + self.spinBox_first.value()) * self.block_size + round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size)] * (
                self.number_of_used_projections)
            print(len(center_list))
            print(center_list)
            print('printing')


            #reconstruct
            if self.algorithm_list.currentText() == 'FBP_CUDA':
                options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
                slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm=tomopy.astra,
                                      options=options)
            else:
                slices = tomopy.recon(extended_sinos, new_list, center=center_list,
                                      algorithm=self.algorithm_list.currentText(),
                                      filter_name=self.filter_list.currentText())

            # scale with pixel size to attenuation coefficients
            slices = slices * (10000 / self.pixel_size.value())

            #crop reconstructed data

            if self.comboBox_180_360.currentText() == '180 - axis centered':
                slices = slices[:, round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size / 2): -round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size / 2),round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size / 2): -round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size / 2)]
            else:
                slices = slices[:, round((self.extend_FOV_fixed_ImageJ_Stream - 1) * self.full_size): -round((self.extend_FOV_fixed_ImageJ_Stream - 1) * self.full_size),round((self.extend_FOV_fixed_ImageJ_Stream - 1) * self.full_size): -round((self.extend_FOV_fixed_ImageJ_Stream - 1) * self.full_size)]

            #slices = slices[:, round(self.extend_FOV * self.full_size /2): -round(self.extend_FOV * self.full_size /2), round(self.extend_FOV * self.full_size /2): -round(self.extend_FOV * self.full_size /2)]
            slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)

            original_reconstruction = slices[:, self.spinBox_top.value():-self.spinBox_bottom.value() - 1,self.spinBox_left.value():-self.spinBox_right.value() - 1]
            print('original_reconstruction.shape', original_reconstruction.shape)

            #16-bit integer conversion
            if self.radioButton_16bit_integer.isChecked() == True:
                ima3 = 65535 * (original_reconstruction - self.int_low.value()) / (self.int_high.value() - self.int_low.value())
                ima3 = numpy.clip(ima3, 1, 65534)
                slices_save = ima3.astype(numpy.uint16)

            if self.radioButton_32bit_float.isChecked() == True:
                slices_save = original_reconstruction

            print('Reconstructed Volume is', slices_save.shape)


            #save data
            if self.save_tiff.isChecked() == True:

                # write the reconstructed block to disk as TIF-file
                a = 1
                while (a < self.block_size + 1) and (a < slices_save.shape[0] + 1):
                    self.progressBar.setValue((a + (i * self.block_size)) * 100 / (self.spinBox_last.value() - self.spinBox_first.value()))
                    QtCore.QCoreApplication.processEvents()
                    time.sleep(0.02)
                    filename2 = self.path_out_reconstructed_full + self.namepart + '_' + str(
                        a + self.crop_offset + i * self.block_size).zfill(4) + '.tif'
                    print('Writing Reconstructed Slices:', filename2)
                    slice_save = slices_save[a - 1, :, :]
                    img = Image.fromarray(slice_save)
                    img.save(filename2)
                    a = a + 1

            if self.save_hdf5.isChecked() == True:
                if i == 0:
                    # create an hdf5-file and write the first reconstructed block into it
                    print(self.path_klick)
                    with h5py.File(self.path_klick, 'r') as f1, h5py.File(self.path_out_reconstructed_full + '/' + self.folder_name + '_reco' + '.h5', 'w') as f:
                        #f.create_dataset("Volume", data=slices_save, chunks = (1,self.hdf_chunking_y.value(),self.hdf_chunking_x.value()), maxshape=(min(slices_save.shape[0],self.spinBox_last.value()-self.spinBox_first.value()), slices_save.shape[1], slices_save.shape[2]))
                        self.hdf_chunking_size_x = math.ceil(slices_save.shape[2]/self.hdf_chunking_x.value())
                        self.hdf_chunking_size_y = math.ceil(slices_save.shape[1]/self.hdf_chunking_y.value())
                        f.create_dataset("Volume", dtype='uint16', data=slices_save, chunks = (self.block_size,self.hdf_chunking_size_y,self.hdf_chunking_size_x), maxshape=(None, slices_save.shape[1], slices_save.shape[2]))
                        f.create_group('/raw_data/instrument')
                        f1.copy('/entry/instrument/NDAttributes', f['/raw_data/instrument'])
                        f1.copy('/entry/instrument/performance', f['/raw_data/instrument'])

                else:
                    # write the subsequent blocks into the hdf5-file
                    self.progressBar.setValue((i * self.block_size) * 100 / (self.spinBox_last.value() - self.spinBox_first.value()))
                    QtCore.QCoreApplication.processEvents()
                    time.sleep(0.02)
                    f = h5py.File(self.path_out_reconstructed_full + '/' + self.folder_name + '_reco' + '.h5', 'r+')
                    vol_proxy_save = f['Volume']
                    vol_proxy_save.resize((vol_proxy_save.shape[0] + slices_save.shape[0]), axis=0)
                    vol_proxy_save[i * self.block_size : i * self.block_size + slices_save.shape[0] ,:,:] = slices_save
                    print(vol_proxy_save.dtype)
                    print('volume_proxy_save.shape', vol_proxy_save.shape)
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
        self.pushReconstruct.setText('Test')
        self.pushReconstruct_all.setText('Reconstruct\n Volume')
        print('Done!')


#no idea why we need this, but it wouldn't work without it ;-)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    main = On_the_fly_CT_tester()
    main.show()
    sys.exit(app.exec_())

#end of code