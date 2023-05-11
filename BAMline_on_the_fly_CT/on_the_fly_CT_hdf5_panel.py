# On-the-fly-CT Reco
version =  "Version 2022.04.20 b"

#Install ImageJ-PlugIn: EPICS AreaDetector NTNDA-Viewer, look for the channel specified here under channel_name, consider multiple users on servers!!!
channel_name = 'BAMline:CTReco'
channel_sino = 'BAMline:CTReco:sino'
standard_path = "B:\\BAMline-CT\\2023\\2023_01\\daisychain_sint_lam_L3_1" # '/mnt/raid/CT/2022/'

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUiType
from PIL import Image
import h5py
import tomopy
import math
import time
import os
import csv
from scipy.ndimage import gaussian_filter, median_filter, shift
from scipy.special import erf
import pvaccess as pva      #to install search for "pvapy"
from output import Ui_MainWindow
from pathlib import Path
from signal_aligment import *
import algotom.prep.removal as rem
from skimage.transform import 


import matplotlib.pyplot as plt

#Ui_on_the_fly_Window, Q_on_the_fly_Window = loadUiType('on_the_fly_CT_reco_hdf_panel.ui')  # connect to the GUI for the program
#how do i load the ui file from the output.py file?


#help me setup ui with the Ui_MainWindow class from the output.py file
#class On_the_fly_CT_tester(Ui_on_the_fly_Window, Q_on_the_fly_Window):
#class On_the_fly_CT_tester(Ui_MainWindow, Q_on_the_fly_Window):
#

class On_the_fly_CT_tester(QtWidgets.QMainWindow, Ui_MainWindow):


    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
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
        self.HDR_compute_button.clicked.connect(self.HDR)
        #self.SampletableWidget.itemClicked.connect(self.update_dataset)
        self.HDR_checkBox.toggled.connect(self.HDR)
        self.HDRnorm = False
        self.pushReconstruct.clicked.connect(self.load)


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
        self.pv_sino = pva.PvObject(pva_image_dict)
        self.pvaServer2 = pva.PvaServer(channel_sino, self.pv_sino)
        self.pvaServer2.start()

    def check(self):
         self.HDRnorm= False
         if self.auto_update.isChecked():
             self.check_test_button()
         return


    def check_test_button(self):

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


    def buttons_activate_crop_volume(self):
        self.spinBox_first.setEnabled(True)
        self.spinBox_last.setEnabled(True)
        self.push_Crop_volume.setEnabled(True)


    #asks the user for a file and loads it in the Sampletablewidget without removing the old entries
    def set_path(self):
        #grey out the buttons while program is busy
        self.buttons_deactivate_all()
        self.pushReconstruct.setText('Busy')
        self.pushReconstruct_all.setText('Busy\n')
        self.new = 1
        self.extend_FOV_fixed_ImageJ_Stream = 0.25

        #ask for hdf5-file
        self.path_klick = QtWidgets.QFileDialog.getOpenFileNames(None, 'Select HDF5 file(s)', standard_path, 'HDF5 files (*.h5 *.hdf5)')[0]
        if self.path_klick == []:
            print('no file selected')
            self.buttons_activate_load()
            self.buttons_activate_reco()
            self.buttons_activate_reco_all()
            return -1
        self.path_klick = [r'D:\meddah\isi_L3_1\1073_230414_0158_isi_L3_1_tomo___Z5_Y7090_50000eV_10x_750ms\1073_230414_0158_00001.h5',
                           r'D:\meddah\isi_L3_1\1079_230414_0451_isi_L3_1_tomo___Z5_Y7090_40000eV_10x_500ms\1079_230414_0451_00001.h5',
                           r'D:\meddah\isi_L3_1\1085_230414_0653_isi_L3_1_tomo___Z5_Y7090_30000eV_10x_300ms\1085_230414_0653_00001.h5']
        print('path klicked: ', self.path_klick)
        self.SampletableWidget.setRowCount(self.path_klick.__len__())
        self.SampletableWidget.setColumnCount(3)
        self.SampletableWidget.setHorizontalHeaderLabels(['Sample','COR','Energy [keV]','Size'])

        self.h5_files = []
        self.COR_files = np.empty(self.path_klick.__len__(), dtype=float)

        for n,path in enumerate(self.path_klick):
            print('chopped path: ', *Path(path).parts,sep=', ')
            #self.SampletableWidget.setItem(n,0,QtWidgets.QTableWidgetItem(Path(path).parts[-1]))
            self.SampletableWidget.setItem(n,0,QtWidgets.QTableWidgetItem(path))
            self.SampletableWidget.setItem(n,3,QtWidgets.QTableWidgetItem(str(h5py.File(path, 'r')['/entry/data/data'].shape)))
            #self.SampletableWidget.setItem(n,2,QtWidgets.QTableWidgetItem(int(h5py.File(path, 'r')['/entry/instrument/NDAttributes/DMM_Energy'][0])))
            # Append the file to the array
            self.h5_files.append(h5py.File(path, 'r'))          

        #self.SampletableWidget.setItemDelegateForColumn(1, self.delegate)  

        #ungrey the buttons for further use of the program
        self.buttons_activate_load()
        self.buttons_activate_reco()
        #self.buttons_activate_crop_volume()
        self.buttons_activate_reco_all()
        self.pushReconstruct.setText('Test')
        self.pushReconstruct_all.setText('Reconstruct\n Volume')

        return -1
    
    def update_dataset(self, item):
        print('Selected dataset: ', item.text())
        #grey out the buttons while program is busy
        self.buttons_deactivate_all()
        self.pushReconstruct.setText('Busy')
        self.pushReconstruct_all.setText('Busy\n')

        #link a volume to the hdf-file
        f = h5py.File(item.text(), 'r')
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
        self.line_proxy = f['/entry/instrument/NDAttributes/CT_MICOS_W']
        print('self.line_proxy', self.line_proxy)
        self.graph = np.array(self.line_proxy[self.spinBox_number_FFs.value(): -self.spinBox_number_FFs.value()])
        print('found number of angles:  ', self.graph.shape, '      angles: ', self.graph)

        #find rotation start
        i = 0
        while i < self.graph.shape[0]:
            if round(self.graph[i]) == 0:  # notice the last projection at below 0.5°
                self.last_zero_proj = i + 3  # assumes 3 images for speeding up the motor
            i = i + 1

        print('Last projection at 0 degree/still speeding up: number', self.last_zero_proj)

        if self.COR.value() == 0:
            self.COR.setValue(round(self.vol_proxy.shape[2] / 2))
            print('COR.setValue', round(self.vol_proxy.shape[2] / 2))

        self.load()


    def change_scan_type(self):
        self.new = 1

        if self.comboBox_180_360.currentText() == '180 - axis centered':
            self.extend_FOV_fixed_ImageJ_Stream = 0.15
        else:
            self.extend_FOV_fixed_ImageJ_Stream = 1.15
        print('extend FOV',self.extend_FOV_fixed_ImageJ_Stream)

        self.check()

    def HDR(self):
        #grey out the buttons while program is busy
        self.buttons_deactivate_all()
        self.pushReconstruct.setText('Busy')
        print(self.SampletableWidget.selectedItems()[0].text())
        if len(self.SampletableWidget.selectedItems()) == 2:
            self.HDR_epsilon()
        elif len(self.SampletableWidget.selectedItems()) == 3:
            self.HDR_biepsilon()
    
    def HDR_epsilon(self):
        print(self.SampletableWidget.selectedItems()[0].text())
        print(self.SampletableWidget.selectedItems()[1].text())
        self.HDRfirst = h5py.File(self.SampletableWidget.selectedItems()[0].text(),'r')
        self.HDRsecond = h5py.File(self.SampletableWidget.selectedItems()[1].text(),'r')

        self.slice_number.setMaximum(self.HDRfirst['/entry/data/data'].shape[1]-1)
        print('set Maximum')
        self.slice_number.setMinimum(0)
        print('set Minimum')
        time.sleep(1)
        self.slice_number.setValue(round(self.HDRfirst['/entry/data/data'].shape[1]/2))    # be careful with an infinite loop when setValue actually triggers valueChanged. Therefore, auto_update starts off
        print('set middle height as slice number:  ', self.slice_number.value())
        self.slice_number.setEnabled(True)

        if self.HDRnorm == False:
            QtWidgets.QApplication.processEvents()
            self.HDRfirst_proxy = self.HDRfirst['/entry/data/data'][self.spinBox_number_FFs.value() : -self.spinBox_number_FFs.value(), self.slice_number.value(), :]
            self.HDRsecond_proxy = self.HDRsecond['/entry/data/data'][self.spinBox_number_FFs.value() : -self.spinBox_number_FFs.value(), self.slice_number.value(), :]
            #
            # 
            #  the smaller image to the size of the larger image, and shift it to align the two images
            if self.HDRfirst_proxy.shape[0] > self.HDRsecond_proxy.shape[0]:
                self.sinoshift = chisqr_align(self.HDRfirst_proxy[:,round(self.HDRfirst_proxy.shape[1]/3)], 
                                          self.HDRsecond_proxy[:,round(self.HDRsecond_proxy.shape[1]/3)], bound=50)
                self.HDRsecond_proxy = shift(
                    
                    (self.HDRsecond_proxy, self.HDRfirst_proxy.shape,anti_aliasing=True), (self.sinoshift, 0))
                print('sinoshift: ', self.sinoshift)
                #get rotation angles
                self.line_proxy = self.HDRfirst['/entry/instrument/NDAttributes/CT_MICOS_W']
                print('self.line_proxy', self.line_proxy)
                self.graph = np.array(self.line_proxy[self.spinBox_number_FFs.value(): -self.spinBox_number_FFs.value()])
                print('found number of angles:  ', self.graph.shape, '      angles: ', self.graph)
                #find rotation start
                i = 0
                while i < self.graph.shape[0]:
                    if round(self.graph[i]) == 0:  # notice the last projection at below 0.5°
                        self.last_zero_proj = i + 3  # assumes 3 images for speeding up the motor
                    i = i + 1

                print('Last projection at 0 degree/still speeding up: number', self.last_zero_proj)

                if self.COR.value() == 0:
                    self.COR.setValue(round(self.HDRfirst_proxy.shape[0] / 2))
                    print('COR.setValue', round(self.HDRfirst_proxy.shape[0] / 2))

            elif self.HDRfirst_proxy.shape[0] < self.HDRsecond_proxy.shape[0]:
                self.sinoshift = chisqr_align(self.HDRsecond_proxy[:,round(self.HDRsecond_proxy.shape[1]/3)],
                                            self.HDRfirst_proxy[:,round(self.HDRfirst_proxy.shape[1]/3)], bound=50)
                self.HDRfirst_proxy = shift(
                    
                    (self.HDRfirst_proxy, self.HDRsecond_proxy.shape,anti_aliasing=True), (self.sinoshift, 0))
                print('sinoshift: ', self.sinoshift)
                #get rotation angles
                self.line_proxy = self.HDRsecond['/entry/instrument/NDAttributes/CT_MICOS_W']
                print('self.line_proxy', self.line_proxy)
                self.graph = np.array(self.line_proxy[self.spinBox_number_FFs.value(): -self.spinBox_number_FFs.value()])
                print('found number of angles:  ', self.graph.shape, '      angles: ', self.graph)
                #find rotation start
                i = 0
                while i < self.graph.shape[0]:
                    if round(self.graph[i]) == 0:  # notice the last projection at below 0.5°
                        self.last_zero_proj = i + 3  # assumes 3 images for speeding up the motor
                    i = i + 1

                print('Last projection at 0 degree/still speeding up: number', self.last_zero_proj)

                if self.COR.value() == 0:
                    self.COR.setValue(round(self.HDRfirst_proxy.shape[0] / 2))
                    print('COR.setValue', round(self.HDRfirst_proxy.shape[0] / 2))


            print('HDRfirst_proxy.shape: ', self.HDRfirst_proxy.shape)
            print('HDRsecond_proxy.shape: ', self.HDRsecond_proxy.shape)
            
            # self.HDRsecond_proxy = np.roll(self.HDRsecond_proxy[self.size_difference_two:,:], self.sinoshift, axis=0)
            self.HDRfirst_proxy_norm = shift(self.HDRfirst_proxy/np.max(self.HDRfirst_proxy),self.sinoshift)
            print('HDRfirst_proxy_norm.shape: ', self.HDRfirst_proxy_norm.shape)
            self.HDRnorm = True
        
        # 16-bit integer conversion
        self.deltafunction = 0.5 - self.epsilon_slider.value()/100 * erf(self.tau_slider.value() * (self.HDRfirst_proxy_norm - self.lambda_slider.value()/100))
        self.SinoHDR = self.deltafunction*self.HDRfirst_proxy + self.HDRsecond_proxy*(1-self.deltafunction)
        #self.Merged_intensities_plot.plot(self.HDRfirst_proxy_norm.flatten(), self.deltafunction.flatten(), symbol='o', clear=True)


        self.Norm = self.SinoHDR
        print('Norm shape', self.Norm.shape, self.Norm[100,100])
        self.spinBox_DF_in_ram = self.spinBox_DF.value()
        self.spinBox_back_illumination_in_ram = self.spinBox_back_illumination.value()
        self.slice_in_ram = self.slice_number.value()
        self.ringradius_in_RAM = self.spinBox_ringradius.value()
        print('set Slice in RAM')
        self.w = self.graph     #no need to load the angles each time a new slice is picked
        print('set angles in self.w')
        

        self.buttons_activate_load()
        self.buttons_activate_reco()
        #self.buttons_activate_crop_volume()
        self.buttons_activate_reco_all()
        self.load()
    
    #function when we have 3 images to merge
    def HDR_biepsilon(self):
        print(self.SampletableWidget.selectedItems()[0].text())
        print(self.SampletableWidget.selectedItems()[1].text())
        print(self.SampletableWidget.selectedItems()[2].text())
        self.HDRfirst = h5py.File(self.SampletableWidget.selectedItems()[0].text(),'r')
        self.HDRsecond = h5py.File(self.SampletableWidget.selectedItems()[1].text(),'r')
        self.HDRthird = h5py.File(self.SampletableWidget.selectedItems()[2].text(),'r')

        self.slice_number.setMaximum(self.HDRfirst['/entry/data/data'].shape[1]-1)
        print('set Maximum')
        self.slice_number.setMinimum(0)
        print('set Minimum')
        time.sleep(1)
        self.slice_number.setValue(round(self.HDRthird['/entry/data/data'].shape[1]/2))    # be careful with an infinite loop when setValue actually triggers valueChanged. Therefore, auto_update starts off
        print('set middle height as slice number:  ', self.slice_number.value())
        self.slice_number.setEnabled(True)

        if self.HDRnorm == False:
            QtWidgets.QApplication.processEvents()
            self.HDRfirst_proxy = self.HDRfirst['/entry/data/data'][self.spinBox_number_FFs.value() : -self.spinBox_number_FFs.value(), self.slice_number.value(), :]
            self.HDRsecond_proxy = self.HDRsecond['/entry/data/data'][self.spinBox_number_FFs.value() : -self.spinBox_number_FFs.value(), self.slice_number.value(), :]
            self.HDRthird_proxy = self.HDRthird['/entry/data/data'][self.spinBox_number_FFs.value() : -self.spinBox_number_FFs.value(), self.slice_number.value(), :]
            #
            # 
            #  the smaller image to the size of the larger image, and shift it to align the two images
            
            #use hdrthird as reference and shift hdrfirst and hdrsecond to hdrthird and 
            # 
            #  them to hdrthird

            self.sino2shift = chisqr_align(self.HDRthird_proxy[:,round(self.HDRthird_proxy.shape[1]/3)], 
                                      self.HDRsecond_proxy[:,round(self.HDRsecond_proxy.shape[1]/3)], bound=50)
            self.HDRsecond_proxy = shift(
                
                (self.HDRsecond_proxy, self.HDRthird_proxy.shape,anti_aliasing=True), (self.sino2shift, 0))
            print('sino2shift', self.sino2shift)

            self.sino1shift = chisqr_align(self.HDRthird_proxy[:,round(self.HDRthird_proxy.shape[1]/3)], 
                                      self.HDRfirst_proxy[:,round(self.HDRfirst_proxy.shape[1]/3)], bound=50)
            self.HDRfirst_proxy = shift(
                
                (self.HDRfirst_proxy, self.HDRthird_proxy.shape,anti_aliasing=True), (self.sino1shift, 0))
            print('sino1shift', self.sino1shift)

            #get rotation angles
            self.line_proxy = self.HDRthird['/entry/instrument/NDAttributes/CT_MICOS_W']
            print('self.line_proxy', self.line_proxy)
            self.graph = np.array(self.line_proxy[self.spinBox_number_FFs.value(): -self.spinBox_number_FFs.value()])
            print('found number of angles:  ', self.graph.shape, '      angles: ', self.graph)
            self.HDRnorm = True

            #find rotation start
            i = 0
            while i < self.graph.shape[0]:
                if round(self.graph[i]) == 0:  # notice the last projection at below 0.5°
                    self.last_zero_proj = i + 3  # assumes 3 images for speeding up the motor
                i = i + 1
            print('Last projection at 0 degree/still speeding up: number', self.last_zero_proj)
            if self.COR.value() == 0:
                self.COR.setValue(round(self.HDRfirst_proxy.shape[0] / 2))
                print('COR.setValue', round(self.HDRfirst_proxy.shape[0] / 2))
                 
            print('HDRfirst_proxy.shape: ', self.HDRfirst_proxy.shape)
            print('HDRsecond_proxy.shape: ', self.HDRsecond_proxy.shape)
            print('HDRthird_proxy.shape: ', self.HDRthird_proxy.shape)
            
            # self.HDRsecond_proxy = np.roll(self.HDRsecond_proxy[self.size_difference_two:,:], self.sinoshift, axis=0)
            #normalize hdrthird_proxy to 1  
            self.HDRthird_proxy_norm = self.HDRthird_proxy/np.max(self.HDRthird_proxy)
            print('HDRfirst_proxy_norm.shape: ', self.HDRthird_proxy_norm.shape)
            self.HDRnorm = True
        
        # 16-bit integer conversion
        self.deltafunction1 = 0.5 - self.epsilon_slider.value()/100 * erf(self.tau_slider.value() * (self.HDRthird_proxy_norm - self.lambda_slider.value()/100))
        self.deltafunction2 = 0.5 + self.epsilon2_slider.value()/100 * erf(self.tau2_slider.value() * (self.HDRthird_proxy_norm - self.lambda2_slider.value()/100))
        self.SinoHDR = self.deltafunction2*self.HDRfirst_proxy + self.HDRsecond_proxy*(1-self.deltafunction1-self.deltafunction2)+ self.HDRthird_proxy*self.deltafunction1
        #self.Merged_intensities_plot.plot(self.HDRfirst_proxy_norm.flatten(), self.deltafunction.flatten(), symbol='o', clear=True)

        self.Norm = self.SinoHDR
        print('Norm shape', self.Norm.shape, self.Norm[100,100])
        self.spinBox_DF_in_ram = self.spinBox_DF.value()
        self.spinBox_back_illumination_in_ram = self.spinBox_back_illumination.value()
        self.slice_in_ram = self.slice_number.value()
        self.ringradius_in_RAM = self.spinBox_ringradius.value()
        print('set Slice in RAM')
        self.w = self.graph     #no need to load the angles each time a new slice is picked
        print('set angles in self.w')
        

        self.buttons_activate_load()
        self.buttons_activate_reco()
        #self.buttons_activate_crop_volume()
        self.buttons_activate_reco_all()
        self.load()


    def load(self):
        self.buttons_deactivate_all()
        if self.HDR_checkBox.isChecked()==False:
            FFs = self.vol_proxy[0:self.spinBox_number_FFs.value() -1, self.slice_number.value(), :]
            FFmean = np.mean(FFs, axis=0)
            print('FFs for normalization ', self.spinBox_number_FFs.value(), FFmean.shape)
            Sino = self.vol_proxy[self.spinBox_number_FFs.value() : -self.spinBox_number_FFs.value(), self.slice_number.value(), :]
            self.Norm = np.divide(np.subtract(Sino, self.spinBox_DF.value()), np.subtract(FFmean, self.spinBox_DF.value() + self.spinBox_back_illumination.value()))
            #self.Norm = self.SinoHDR
            print('Norm shape', self.Norm.shape, self.Norm[100,100])
            self.spinBox_DF_in_ram = self.spinBox_DF.value()
            self.spinBox_back_illumination_in_ram = self.spinBox_back_illumination.value()
            self.slice_in_ram = self.slice_number.value()
            self.ringradius_in_RAM = self.spinBox_ringradius.value()
            print('set Slice in RAM')
            self.w = self.graph     #no need to load the angles each time a new slice is picked
            print('set angles in self.w')
            
        #Ring artifact handling
        if self.spinBox_ringradius.value() != 0:
            self.proj_sum = np.mean(self.Norm, axis = 0)
            self.proj_sum_2d = np.zeros((1, self.proj_sum.shape[0]), dtype = np.float32)
            self.proj_sum_2d[0,:] = self.proj_sum
            print('proj_sum dimensions', self.proj_sum.shape)
            print('proj_sum_2d dimensions', self.proj_sum_2d.shape)

            proj_sum_filtered = median_filter(self.proj_sum_2d, size = (1,self.spinBox_ringradius.value()), mode='nearest')
            print('proj_sum_filtered dimensions', proj_sum_filtered.shape)
            correction_map = np.divide(self.proj_sum_2d, proj_sum_filtered)
            correction_map = np.clip(correction_map, 0.9, 1.1)
            print('correction_map dimensions', correction_map.shape, 'correction_map min vs max', np.amin(correction_map), np.amax(correction_map))

            i=0
            while i < self.Norm.shape[0]:
                self.Norm[i, :] = np.divide(self.Norm[i, :], correction_map[0,:])
                self.progressBar.setValue((i + 1) * 100 / self.Norm.shape[0])
                QtWidgets.QApplication.processEvents()
                i = i+1
            print('Norm.shape', self.Norm.shape)
            print('finished ring handling')
        else:
            print('did not do ring handling')

        #prefill cropping
        self.spinBox_first.setValue(0)
        self.spinBox_last.setValue(self.Norm.shape[1]-1)
        print('set possible crop range')

        #prefill rotation-speed[°/img]
        #Polynom fit for the angles
        poly_coeff = np.polyfit(np.arange(len(self.w[round((self.w.shape[0] + 1) /4) : round((self.w.shape[0] + 1) * 3/4) ])), self.w[round((self.w.shape[0] + 1) /4) : round((self.w.shape[0] + 1) * 3/4) ], 1, rcond=None, full=False, w=None, cov=False)
        print('Polynom coefficients',poly_coeff, '   Detected angular step per image: ', poly_coeff[0])
        self.speed_W.setValue(poly_coeff[0])
        print('Last projection at 0 degree/still speeding up: image number', self.last_zero_proj)

        time.sleep(1) #???

        #ungrey the buttons for further use of the program
        self.buttons_activate_load()
        self.buttons_activate_reco()
        #self.buttons_activate_crop_volume()
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
        if self.number_of_projections * self.speed_W.value() >= 270:
            self.number_of_used_projections = round(360 / self.speed_W.value())
        else:
            #print('smaller than 3/2 Pi')
            self.number_of_used_projections = round(180 / self.speed_W.value())
        print('number of projections used for reconstruction (omitting those above 180°/360°: )', self.number_of_used_projections)

        # create list with all projection angles
        new_list = (np.arange(self.number_of_used_projections) * self.speed_W.value() + self.Offset_Angle.value()) * math.pi / 180

        # create list with x-positions of projections
        if self.comboBox_180_360.currentText() == '360 - axis right':
            center_list = [self.COR.value() + round((self.extend_FOV_fixed_ImageJ_Stream -1) * self.full_size)] * (self.number_of_used_projections)
            #center_list = [self.COR.value() +  self.full_size] * (self.number_of_used_projections)
        else:
            center_list = [self.COR.value() + round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size)] * (self.number_of_used_projections)

        # create one sinogram in the form [z, y, x]
        transposed_sinos = np.zeros((min(self.number_of_used_projections, self.Norm.shape[0]), 1, self.full_size), dtype=float)
        transposed_sinos[:,0,:] = self.Norm[self.last_zero_proj : min(self.last_zero_proj + self.number_of_used_projections, self.Norm.shape[0]),:]
        

        #extend data with calculated parameter, compute logarithm, remove NaN-values
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

        extended_sinos = np.nan_to_num(extended_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
        print('extended_sinos.shape', extended_sinos.shape)

        #apply phase retrieval if desired
        if self.checkBox_phase_2.isChecked() == True:
            extended_sinos = tomopy.prep.phase.retrieve_phase(extended_sinos, pixel_size=0.0001, dist=self.doubleSpinBox_distance_2.value(), energy=self.doubleSpinBox_Energy_2.value(), alpha=self.doubleSpinBox_alpha_2.value(), pad=True, ncore=None, nchunk=None)
            print('applying phase contrast')
        
        extended_sinos[:,0,:] = rem.remove_all_stripe(extended_sinos[:,0,:], snr=3, la_size=51,sm_size=21)


        if isinstance(transposed_sinos[0,0,0],np.integer):
            self.typesinosent = 'ushortValue'
        elif isinstance(transposed_sinos[0,0,0],np.float32):
            self.typesinosent = 'floatValue'
        elif isinstance(transposed_sinos[0,0,0],np.float64):
            self.typesinosent = 'doubleValue'
        else:
            print('Datatype unknown')
            return

        # set image dimensions only for the first time or when scan-type was changed
        if self.new == 1:
            self.pv_sino['dimension'] = [
                {'size': transposed_sinos.shape[2], 'fullSize': transposed_sinos.shape[2], 'binning': 1},
                {'size': transposed_sinos.shape[0], 'fullSize': transposed_sinos.shape[0], 'binning': 1}]
        
        self.pv_sino['value'] = ({self.typesinosent: transposed_sinos[:,0,:].flatten()},)

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

        slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)
        original_reconstruction = slices[0, :, :]
        print('original_reconstruction.shape', original_reconstruction.shape)

        if isinstance(original_reconstruction[0,0],np.integer):
            self.typesent = 'ushortValue'
        elif isinstance(original_reconstruction[0,0],np.float32):
            self.typesent = 'floatValue'
        elif isinstance(original_reconstruction[0,0],np.float64):
            self.typesent = 'doubleValue'
        else:
            print('Datatype unknown')
            return

        # set image dimensions only for the first time or when scan-type was changed
        if self.new == 1:
            self.pv_rec['dimension'] = [
                {'size': original_reconstruction.shape[0], 'fullSize': original_reconstruction.shape[0], 'binning': 1},
                {'size': original_reconstruction.shape[0], 'fullSize': original_reconstruction.shape[0], 'binning': 1}]
            self.new = 0
        


        # 16-bit integer conversion
        if self.radioButton_16bit_integer.isChecked() == True:
            self.ima3 = 65535 * (original_reconstruction - self.int_low.value()) / (self.int_high.value() - self.int_low.value())
            self.ima3 = np.clip(self.ima3, 1, 65534)
            self.ima3 = np.around(self.ima3)
            print('slice_show.shape',self.ima3.shape)

            # write result to pv
            self.pv_rec['value'] = ({self.typesent: self.ima3.flatten()},)
            print('Reconstructed Slice is', self.ima3.shape)

        # keep 32-bit float
        if self.radioButton_32bit_float.isChecked() == True:
            self.slice_show = original_reconstruction

            # write result to pv
            self.pv_rec['value'] = ({self.typesent: self.slice_show.flatten()},)
            print('Reconstructed Slice is', self.slice_show.shape)


        #find and display minimum and maximum values in reconstructed slice
        print('minimum value found: ', np.amin(original_reconstruction), '     maximum value found: ',np.amax(original_reconstruction))

        print('reconstruction of slice is done')

        #ungrey the buttons for further use of the program
        self.buttons_activate_load()
        self.buttons_activate_reco()
        #self.buttons_activate_crop_volume()
        self.buttons_activate_reco_all()
        self.pushReconstruct.setText('Test')
        self.pushReconstruct_all.setText('Reconstruct\n Volume')



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
            self.path_out_reconstructed_full = self.path_out_reconstructed_ask + '/'+ self.folder_name
            os.mkdir(self.path_out_reconstructed_full)
        if self.save_hdf5.isChecked() == True:
            self.path_out_reconstructed_full = self.path_out_reconstructed_ask

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
        new_list = (np.arange(self.number_of_used_projections) * self.speed_W.value() + self.Offset_Angle.value()) * math.pi / 180
        print(new_list.shape)

        # create list with COR-positions
        center_list = [self.COR.value() + round(self.extend_FOV * self.full_size)] * (self.number_of_used_projections)
        print(len(center_list))

        #save parameters in csv-file
        file_name_parameter = self.path_out_reconstructed_full + '/' + self.folder_name + '_parameter.csv'
        with open(file_name_parameter, mode = 'w', newline='') as parameter_file:
            csv_writer = csv.writer(parameter_file, delimiter = '\t', quotechar=' ')
            csv_writer.writerow(['Path input                    ', self.path_in,' '])
            csv_writer.writerow(['Path output                   ', self.path_out_reconstructed_full,' '])
            csv_writer.writerow(['Number of used projections    ', str(self.number_of_used_projections),' '])
            csv_writer.writerow(['Center of rotation            ', str(self.COR.value()), ' '])
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
        i = 0
        while (i < math.ceil(self.vol_proxy.shape[1] / self.block_size)):       #Need to fix the rest of slices

            print('Reconstructing block', i + 1, 'of', math.ceil(self.vol_proxy.shape[1] / self.block_size))

            FFs_vol = self.vol_proxy[0:self.spinBox_number_FFs.value() - 1, i * self.block_size: (i + 1) * self.block_size, :]
            FFmean_vol = np.mean(FFs_vol, axis=0)
            print('FFs for normalization ', self.spinBox_number_FFs.value(), FFmean_vol.shape)
            Sino_vol = self.vol_proxy[self.spinBox_number_FFs.value(): -self.spinBox_number_FFs.value(), i * self.block_size: (i + 1) * self.block_size, :]
            self.Norm_vol = np.divide(Sino_vol - self.spinBox_DF.value(), FFmean_vol - self.spinBox_DF.value() - self.spinBox_back_illumination.value())
            print('sinogram shape', self.Norm_vol.shape)

            # Ring artifact handling
            if self.spinBox_ringradius.value() != 0:
                self.proj_sum = np.mean(self.Norm_vol, axis=0)
                print('proj_sum dimensions', self.proj_sum.shape)
                proj_sum_filtered = median_filter(self.proj_sum, size= self.spinBox_ringradius.value(), mode='nearest')
                print('proj_sum_filtered dimensions', proj_sum_filtered.shape)
                correction_map_vol = np.divide(self.proj_sum, proj_sum_filtered)
                correction_map_vol = np.clip(correction_map_vol, 0.9, 1.1)
                print('correction_map_vol dimensions', correction_map_vol.shape, 'correction_map min vs max',
                      np.amin(correction_map_vol), np.amax(correction_map_vol))

                j = 0
                while j < self.Norm_vol.shape[0]:
                    self.Norm_vol[j, :,:] = np.divide(self.Norm_vol[j, :,:], correction_map_vol)
                    j = j + 1
                print('Norm_vol.shape', self.Norm_vol.shape)
                print('finished ring handling')
            else:
                print('did not do ring handling')




            #extend data, take logarithm, remove NaN-values
            extended_sinos = self.Norm_vol[self.last_zero_proj : min(self.last_zero_proj + self.number_of_used_projections, self.Norm_vol.shape[0]), :, :]

            extended_sinos = tomopy.misc.morph.pad(extended_sinos, axis=2, npad=round(self.extend_FOV * self.full_size), mode='edge')
            extended_sinos = tomopy.minus_log(extended_sinos)
            #extended_sinos = (extended_sinos + 9.68) * 1000  # conversion factor to uint
            extended_sinos = np.nan_to_num(extended_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)

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

            # scale with pixel size to attenuation coefficients
            slices = slices * (10000 / self.pixel_size.value())

            #crop reconstructed data
            slices = slices[:, round(self.extend_FOV * self.full_size /2): -round(self.extend_FOV * self.full_size /2), round(self.extend_FOV * self.full_size /2): -round(self.extend_FOV * self.full_size /2)]
            slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)

            #16-bit integer conversion
            if self.radioButton_16bit_integer.isChecked() == True:
                ima3 = 65535 * (slices - self.int_low.value()) / (self.int_high.value() - self.int_low.value())
                ima3 = np.clip(ima3, 1, 65534)
                slices_save = ima3.astype(np.uint16)

            if self.radioButton_32bit_float.isChecked() == True:
                slices_save = slices

            print('Reconstructed Volume is', slices_save.shape)


            #save data
            if self.save_tiff.isChecked() == True:

                # write the reconstructed block to disk as TIF-file
                a = 1
                while (a < self.block_size + 1) and (a < slices_save.shape[0] + 1):
                    self.progressBar.setValue((a + (i * self.block_size)) * 100 / self.vol_proxy.shape[1])
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
                    # create an an hdf5-file and write the first reconstructed block into it
                    with h5py.File(self.path_out_reconstructed_full + '/' + self.folder_name + '.h5', 'w') as f:
                        f.create_dataset("Volume", data=slices_save, chunks = (1,self.hdf_chunking_y.value(),self.hdf_chunking_x.value()), maxshape=(None, slices_save.shape[1], slices_save.shape[2]))
                else:
                    # write the subsequent blocks into the hdf5-file
                    self.progressBar.setValue((i * self.block_size) * 100 / self.vol_proxy.shape[1])
                    QtCore.QCoreApplication.processEvents()
                    time.sleep(0.02)
                    f = h5py.File(self.path_out_reconstructed_full + '/' + self.folder_name + '.h5', 'r+')
                    vol_proxy_save = f['Volume']
                    print('volume_proxy_save.shape', vol_proxy_save.shape)
                    vol_proxy_save.
                    
                    ((vol_proxy_save.shape[0] + slices_save.shape[0]), axis=0)
                    vol_proxy_save[i * self.block_size : i * self.block_size + slices_save.shape[0] ,:,:] = slices_save


            i = i + 1

        #set progress bar to 100%
        self.progressBar.setValue(100)
        QtCore.QCoreApplication.processEvents()
        time.sleep(0.02)

        #ungrey the buttons for further use of the program
        self.buttons_activate_load()
        self.buttons_activate_reco()
        #self.buttons_activate_crop_volume()
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