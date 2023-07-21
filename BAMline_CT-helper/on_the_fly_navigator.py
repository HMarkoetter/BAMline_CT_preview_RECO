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
import cv2                                      #to install package with pycharm search for "opencv-python"
from scipy.ndimage.filters import gaussian_filter, median_filter
import pvaccess as pva                          #to install package with pycharm search for "pvapy"


# On-the-fly Navigator
version =  "Version 2023.07.20 a"

#Install ImageJ-PlugIn: EPICS AreaDetector NTNDA-Viewer, look for the channel specified here under channel_name, consider multiple users on servers!!!
channel_name = 'BAMline:Navigator'
#standard_path = "C:/temp/HDF5-Reading/220130_1734_604_J1_anode_half_cell_in-situ_Z30_Y5430_15000eV_1p44um_500ms/" # '/mnt/raid/CT/2022/'
standard_path = r'C:/delete/reg_data/18_230606_2044_AlTi_F_Ref_tomo___Z25_Y6500_25000eV_10x_400ms'

Ui_on_the_fly_Navigator_Window, Q_on_the_fly_Navigator_Window = loadUiType('on_the_fly_navigator.ui')  # connect to the GUI for the program


class OnTheFlyNavigator(Ui_on_the_fly_Navigator_Window, Q_on_the_fly_Navigator_Window):


    def __init__(self):
        super(OnTheFlyNavigator, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('On-the-fly Navigator')

        #connect buttons to actions
        self.pushLoad.clicked.connect(self.set_path)

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

    def set_path(self):
        print('function set_path')
        #grey out the buttons while program is busy
        #self.buttons_deactivate_all()

        self.new = 1
        self.extend_FOV_fixed_ImageJ_Stream = 1.0
        self.ruler_grid_line_thickness = 4

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
        print('function cut_path_name')
        #analyse and cut the path in pieces and get relevant information from raw-file
        htap = self.path_klick[::-1]
        self.path_in = self.path_klick[0: len(htap) - htap.find('/') - 1: 1]
        ni_htap = self.path_in[::-1]
        self.last_folder = self.path_in[len(ni_htap) - ni_htap.find('/') - 1 :  :1]
        self.namepart = self.path_klick[len(htap) - htap.find('/') - 1: len(htap) - htap.find('.') - 1: 1]
        self.filetype = self.path_klick[len(htap) - htap.find('.') - 1: len(htap):1]
        print('chopped path: ',self.path_in, '  ', self.last_folder,'  ', self.namepart,'  ', self.filetype)
        self.Sample.setText(self.path_klick)

        #link a volume to the hdf-file
        self.f = h5py.File(self.path_klick, 'r')
        self.vol_proxy = self.f['/entry/data/data']
        print('raw data volume size: ', self.vol_proxy.shape)

        self.prefill_parameter()
        #self.buttons_activate_reco()
        print('try to go to function check_180 ')
        self.check_180()    #this will end in check_auto_update



#===========================================================
    def buttons_activate_reco(self):
        print('function buttons_activate_reco')
        #self.slice_number.setEnabled(True)
        #self.COR_1.setEnabled(True)
        #self.COR_2.setEnabled(True)
        #self.COR_3.setEnabled(True)
        #self.COR_4.setEnabled(True)

    def prefill_parameter(self):
        print('function prefill_parameter')
        self.prefill_slice_number()
        self.get_rotation_angles()
        self.find_rotation_start()
        self.prefill_CORs()
        self.prefill_pixel_size()
        self.prefill_energy()
        self.prefill_distance()
        print('done prefill parameter')

    def check_180(self):
        print('function check_180')
        print(self.graph, self.graph[-1])
        while self.graph[-1] < 180:
            print('waiting for sufficient data...', self.graph[-1])
            time.sleep(1)
        print("Enough data to proceed")
        self.check_auto_update()

    def check_auto_update(self):    #AUTO UPDATE ON/OFF?
        print('function check_auto_update')
        i=0
        while i < 1:
            j=0
            print('j:', j)
            j=j+1
            while self.auto_update.isChecked() == False:
                print('waiting for auto update...')
                QtWidgets.QApplication.processEvents()
                time.sleep(5)
            print('auto update requested')
            while self.auto_update.isChecked() == True:
                time_begin = time.time()
                self.read_parameter()
                time_read_parameter = time.time()
                print('time_read_parameter',time_read_parameter - time_begin)
                QtWidgets.QApplication.processEvents()
                self.load_data()
                time_load_data = time.time()
                print('time_load_data', time_load_data - time_begin)
                QtWidgets.QApplication.processEvents()
                self.reconstruct()
                time_reconstruct = time.time()
                print('time_reconstruct', time_reconstruct - time_begin)
                QtWidgets.QApplication.processEvents()

    def read_parameter(self):
        #print('function read_parameter')
        self.prefill_pixel_size()
        self.prefill_energy()
        self.prefill_distance()


    def buttons_deactivate_all(self):
        print('function buttons_deactivate_all')
        self.pushLoad.setEnabled(False)
        self.slice_number.setEnabled(False)
        self.COR_1.setEnabled(False)
        self.COR_2.setEnabled(False)
        self.COR_3.setEnabled(False)
        self.COR_4.setEnabled(False)

    def buttons_activate_load(self):
        print('function buttons_activate_load')
        self.pushLoad.setEnabled(True)


    def prefill_slice_number(self):  #get the image dimensions and prefill slice number
        print('function prefill_slice_number')
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

    def get_rotation_angles(self):
        print('function get_rotation_angles')
        self.line_proxy = self.f['/entry/instrument/NDAttributes/CT_MICOS_W']
        print('self.line_proxy', self.line_proxy)
        self.graph = numpy.array(self.line_proxy[self.spinBox_number_FFs.value(): -self.spinBox_number_FFs.value()])
        print('found number of angles:  ', self.graph.shape[0], '      angles: ', self.graph)

    def find_rotation_start(self):
        print('function find_rotation_start')
        i = 0
        while i < self.graph.shape[0]:
            if round(self.graph[i]) == 0:  # notice the last projection at below 0.5°
                self.last_zero_proj = i + 3  # assumes 3 images for speeding up the motor
            i = i + 1
        print('Last projection at 0 degree/still speeding up: number', self.last_zero_proj)

    def prefill_CORs(self):
        #print('function prefill_CORs')
        if self.COR_1.value() == 0:
            self.COR_1.setValue(round(self.vol_proxy.shape[2] / 2))
            self.COR_2.setValue(round(self.vol_proxy.shape[2] / 2))
            self.COR_3.setValue(round(self.vol_proxy.shape[2] / 2))
            self.COR_4.setValue(round(self.vol_proxy.shape[2] / 2))
            #print('COR.setValue', round(self.vol_proxy.shape[2] / 2))

    def prefill_pixel_size(self):
        #print('function prefill_pixel_size')
        if '/entry/instrument/NDAttributes/CT_Pixelsize' in self.f:
            self.pixel_proxy = self.f['/entry/instrument/NDAttributes/CT_Pixelsize']
            #print('self.pixel_proxy: first ', self.pixel_proxy[0], ' last ', self.pixel_proxy[-1])
            self.pixel_size.setValue(self.pixel_proxy[-1])
        else:
            self.pixel_size.setValue(1)
            #print('pixel size not found')
        self.pixel_size.setEnabled(True)

    def prefill_energy(self):
        #print('function prefill_energy')
        #get and prefill energy
        if '/entry/instrument/NDAttributes/DMM_Energy' in self.f:
            self.energy_proxy = self.f['/entry/instrument/NDAttributes/DMM_Energy']
            #print('self.energy_proxy first ', self.energy_proxy[0], ' last ', self.energy_proxy[-1])
            self.doubleSpinBox_Energy_2.setValue(round(self.energy_proxy[-1]*100)/100)
        else:
            self.doubleSpinBox_Energy_2.setValue(1)
            #print('Energy not found')
        self.doubleSpinBox_Energy_2.setEnabled(True)

    def prefill_distance(self):
        #print('function prefill_distance')
        #get and prefill distance
        if '/entry/instrument/NDAttributes/CT-Kamera-Z' in self.f:
            self.distance_proxy = self.f['/entry/instrument/NDAttributes/CT-Kamera-Z']
            #print('self.distance_proxy first ', self.distance_proxy[0], ' last ', self.distance_proxy[-1])
            #print(round(self.distance_proxy[-1] + 25))
            self.doubleSpinBox_distance_2.setValue(round(self.distance_proxy[-1] + 25))
        else:
            self.doubleSpinBox_distance_2.setValue(0)
            #print('Distance not found')
        self.doubleSpinBox_distance_2.setEnabled(True)
        QtWidgets.QApplication.processEvents()
        #print('prefill distance done')



    def load_data(self):
        #print('function load')
        #self.f = h5py.File(self.path_klick, 'r')
        self.vol_proxy = self.f['/entry/data/data']
        #print('raw data volume size: ', self.vol_proxy.shape)

        #NORMALIZATION TURNED OFF FOR SPEED
        #FFs = self.vol_proxy[0:self.spinBox_number_FFs.value() -1, self.slice_number.value(), :]
        #FFmean = numpy.mean(FFs, axis=0)
        #print('FFs for normalization ', self.spinBox_number_FFs.value(), FFmean.shape)
        Sino = self.vol_proxy[self.spinBox_number_FFs.value() : -self.spinBox_number_FFs.value(), self.slice_number.value(), :]
        #self.Norm = numpy.divide(numpy.subtract(Sino, self.spinBox_DF.value()), numpy.subtract(FFmean, self.spinBox_DF.value() + self.spinBox_back_illumination.value()))
        #self.Norm = numpy.divide(Sino, FFmean)
        self.Norm = Sino
        #print('Norm shape', self.Norm.shape)
        self.w = self.graph
        #print('set angles in self.w')

        """
        #Ring artifact handling turned off for speed reasons
        self.proj_sum = numpy.mean(self.Norm, axis = 0)
        self.proj_sum_2d = numpy.zeros((1, self.proj_sum.shape[0]), dtype = numpy.float32)
        self.proj_sum_2d[0,:] = self.proj_sum
        #print('proj_sum dimensions', self.proj_sum.shape)
        #print('proj_sum_2d dimensions', self.proj_sum_2d.shape)

        proj_sum_filtered = median_filter(self.proj_sum_2d, size = (1,50), mode='nearest')
        #print('proj_sum_filtered dimensions', proj_sum_filtered.shape)
        correction_map = numpy.divide(self.proj_sum_2d, proj_sum_filtered)
        correction_map = numpy.clip(correction_map, 0.5, 2.0)
        #print('correction_map dimensions', correction_map.shape, 'correction_map min vs max', numpy.amin(correction_map), numpy.amax(correction_map))

        i=0
        while i < self.Norm.shape[0]:
            self.Norm[i, :] = numpy.divide(self.Norm[i, :], correction_map[0,:])
            #self.progressBar.setValue((i + 1) * 100 / self.Norm.shape[0])
            #QtWidgets.QApplication.processEvents()
            i = i+1
        #print('Norm.shape', self.Norm.shape)
        #print('finished ring handling')


        #prefill rotation-speed[°/img]
        #Polynom fit for the angles
        """
        poly_coeff = numpy.polyfit(numpy.arange(len(self.w[round((self.w.shape[0] + 1) /4) : round((self.w.shape[0] + 1) * 3/4) ])), self.w[round((self.w.shape[0] + 1) /4) : round((self.w.shape[0] + 1) * 3/4) ], 1, rcond=None, full=False, w=None, cov=False)
        #print('Polynom coefficients',poly_coeff, '   Detected angular step per image: ', poly_coeff[0])
        self.speed_W = poly_coeff[0]

        #time.sleep(1) #???

        #ungrey the buttons for further use of the program
        #self.buttons_activate_load()
        #print('Loading/changing slice complete!')

        #self.reconstruct()

    def add_ruler(self):
        #print('function add_ruler')
        self.f = h5py.File(self.path_klick, 'r')
        if '/entry/instrument/NDAttributes/CT_Piezo_X45' in self.f:
            self.piezo_45_proxy = self.f['/entry/instrument/NDAttributes/CT_Piezo_X45']
            #print('self.piezo_45_proxy: ', self.piezo_45_proxy[0], self.piezo_45_proxy[-1],round(1000 * self.piezo_45_proxy[-1]))
            self.piezo_135_proxy = self.f['/entry/instrument/NDAttributes/CT_Piezo_Y45']
            #print('self.piezo_135_proxy: ', self.piezo_135_proxy[0], self.piezo_135_proxy[-1],round(1000 * self.piezo_135_proxy[-1]))
        else:
            self.piezo_45_proxy = (0,0)
            self.piezo_135_proxy = (0,0)

        if 'pixel_proxy' in globals():
            #print('pixel_proxy', self.pixel_proxy[-1])
            if self.pixel_proxy[-1] == 3.6:
                self.spinBox_ruler_grid = self.spinBox_ruler_grid_1.value()
            elif self.pixel_proxy[-1] == 1.44:
                self.spinBox_ruler_grid = self.spinBox_ruler_grid_2.value()
            elif self.pixel_proxy[-1] == 0.72:
                self.spinBox_ruler_grid = self.spinBox_ruler_grid_3.value()
            elif self.pixel_proxy[-1] == 0.36:
                self.spinBox_ruler_grid = self.spinBox_ruler_grid_4.value()
        else:
            #print('pixel size undefined. Set to spinBox_ruler_grid_1')
            self.spinBox_ruler_grid = self.spinBox_ruler_grid_1.value()
        #print('self.spinBox_ruler_grid', self.spinBox_ruler_grid)

        print('numpy.max', numpy.max(self.slice), round(numpy.max(self.slice)))
        self.ruler_grid_color = math.ceil(numpy.max(self.slice))
        # draws a circle with the detector size as diameter
        cv2.circle(self.slice, (round(self.slice.shape[1] / 2), round(self.slice.shape[1] / 2)), round(self.full_size/2), self.ruler_grid_color, self.ruler_grid_line_thickness)

        if self.grid_micrometer.isChecked() == True:
            print(self.pixel_size.value())
            # add ruler +X
            for r in range(round(self.slice.shape[1] * self.pixel_size.value() / 2),
                           round(self.slice.shape[1] * self.pixel_size.value()),
                           round(self.spinBox_ruler_grid)):
                cv2.line(self.slice, (round(r / self.pixel_size.value()), 0),
                         (round(r / self.pixel_size.value()), self.slice.shape[0]), self.ruler_grid_color, self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round(1000 * self.piezo_45_proxy[-1]/  5) * 5   +   round( r / 5) * 5   -   round(self.slice.shape[1] * (self.pixel_size.value() / 10)) * 5),
                            (round(r / self.pixel_size.value()) + 20, round(self.ruler_grid_line_thickness)*20), cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness/2),
                            self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)

            # add ruler -X
            for r in range(round(self.slice.shape[1] * self.pixel_size.value() / 2), 0,
                           -round(self.spinBox_ruler_grid)):
                cv2.line(self.slice, (round(r / self.pixel_size.value()), 0),
                         (round(r / self.pixel_size.value()), self.slice.shape[0]), self.ruler_grid_color, self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round(1000 * self.piezo_45_proxy[-1] / 5) * 5  +  round(
                    r  / 5) * 5  -  round(self.slice.shape[1] * (self.pixel_size.value() / 10)) * 5),
                            (round(r / self.pixel_size.value()) + 20, round(self.ruler_grid_line_thickness)*20), cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness/2),
                            self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)

            # add ruler +Y
            for r in range(round(self.slice.shape[1] * self.pixel_size.value() / 2),
                           round(self.slice.shape[1] * self.pixel_size.value()),
                           round(self.spinBox_ruler_grid)):
                cv2.line(self.slice, (0, round(r / self.pixel_size.value())),
                         (self.slice.shape[1], round(r / self.pixel_size.value())), self.ruler_grid_color, self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round(1000 * self.piezo_135_proxy[-1] / 5) * 5  +  round(
                    r / 5) * 5 - round(self.slice.shape[1] * (self.pixel_size.value() / 10)) * 5),
                            (20, round(r / self.pixel_size.value()) + round(self.ruler_grid_line_thickness)*20), cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness/2),
                            self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)

            # add ruler -Y
            for r in range(round(self.slice.shape[1] * self.pixel_size.value() / 2), 0,
                           -round(self.spinBox_ruler_grid)):
                cv2.line(self.slice, (0, round(r / self.pixel_size.value())),
                         (self.slice.shape[1], round(r / self.pixel_size.value())), self.ruler_grid_color, self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round(1000 * self.piezo_135_proxy[-1] / 5) * 5 + round(
                    r / 5) * 5 - round(self.slice.shape[1] * (self.pixel_size.value()/ 10)) * 5),
                            (20, round(r / self.pixel_size.value()) + round(self.ruler_grid_line_thickness)*20), cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness/2),
                            self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)

        if self.grid_pixel.isChecked() == True:

            # add ruler +X
            for r in range(round(self.slice.shape[1] / 2), self.slice.shape[1], round(self.spinBox_pixel_grid.value())):
                cv2.line(self.slice, (r, 0),   (r, self.slice.shape[1]), (65535, 65535, 65535), self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (r + 20, round(self.ruler_grid_line_thickness/2)*20), cv2.FONT_HERSHEY_SIMPLEX, self.ruler_grid_line_thickness/2, (65535, 65535, 65535), thickness=self.ruler_grid_line_thickness)

            # add ruler -X
            for r in range(round(self.slice.shape[1] / 2), 0, -round(self.spinBox_pixel_grid.value())):
                #print(r)
                cv2.line(self.slice, (r, 0),   (r, self.slice.shape[1]), (65535, 65535, 65535), self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (r + 20, round(self.ruler_grid_line_thickness/2)*20), cv2.FONT_HERSHEY_SIMPLEX, self.ruler_grid_line_thickness/2, (65535, 65535, 65535), thickness=self.ruler_grid_line_thickness)

            # add ruler +Y
            for r in range(round(self.slice.shape[1] / 2), self.slice.shape[1], round(self.spinBox_pixel_grid.value())):
                cv2.line(self.slice, (0, r), (self.slice.shape[1], r), (65535, 65535, 65535), self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (20,r + round(self.ruler_grid_line_thickness/2)*20), cv2.FONT_HERSHEY_SIMPLEX, self.ruler_grid_line_thickness/2, (65535, 65535, 65535), thickness=self.ruler_grid_line_thickness)

            # add ruler -Y
            for r in range(round(self.slice.shape[1] / 2), 0, -round(self.spinBox_pixel_grid.value())):
                cv2.line(self.slice, (0, r), (self.slice.shape[1], r), (65535, 65535, 65535), self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (20,r + round(self.ruler_grid_line_thickness/2)*20), cv2.FONT_HERSHEY_SIMPLEX, self.ruler_grid_line_thickness/2, (65535, 65535, 65535), thickness=self.ruler_grid_line_thickness)

    def reconstruct(self):
        #print('function reconstruct')

        QtWidgets.QApplication.processEvents()
        #print('def reconstruct')

        self.full_size = self.Norm.shape[1]
        self.number_of_projections = self.Norm.shape[0]

        self.number_of_used_projections = round(180 / self.speed_W)
        #print('number of projections used for reconstruction (omitting those above 180°): ', self.number_of_used_projections)

        # create list with all projection angles
        new_list = (numpy.arange(self.number_of_used_projections) * self.speed_W + self.graph[-1]) * math.pi / 180
        #print('new list in radiant', new_list)

        if 'pixel_proxy' in globals():
            #print('pixel_proxy', self.pixel_proxy[-1])
            if self.pixel_proxy[-1] == 3.6:
                self.COR = self.COR_1.value()
            elif self.pixel_proxy[-1] == 1.44:
                self.COR = self.COR_2.value()
            elif self.pixel_proxy[-1] == 0.72:
                self.COR = self.COR_3.value()
            elif self.pixel_proxy[-1] == 0.36:
                self.COR = self.COR_4.value()
        else:
            #print('pixel size undefined. Set to COR_1')
            self.COR = self.COR_1.value()
        #print('self.COR', self.COR)

        center_list = [self.COR + round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size)] * (self.number_of_used_projections)

        # create one sinogram in the form [z, y, x]
        transposed_sinos = numpy.zeros((self.number_of_used_projections, 1, self.full_size), dtype=float)
        #transposed_sinos[:,0,:] = self.Norm[self.last_zero_proj : min(self.last_zero_proj + self.number_of_used_projections, self.Norm.shape[0]),:]
        transposed_sinos[:,0,:] = self.Norm[-self.number_of_used_projections : , : ]

        #extend data with calculated parameter, compute logarithm, remove NaN-values
        ### cut and reorder 360°-sinos to 180°-sinos, work in progress !!!
        log_sinos = tomopy.minus_log(transposed_sinos)
        log_sinos = numpy.nan_to_num(log_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
        extended_sinos = tomopy.misc.morph.pad(log_sinos, axis=2,
                                               npad=round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size),
                                               mode='edge')

        #print('extended_sinos.shape', extended_sinos.shape)

        #reconstruct one slice
        slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm='gridrec', filter_name='shepp')

        # scale with pixel size to attenuation coefficients
        slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)
        self.slice = slices[0,:,:]   #reduce dimensions from 3 to 2
        #self.slice = self.slice * (10000 / self.pixel_size.value())

        self.add_ruler()

        #crop before sending to ImageJ      spinBox_left
        #original_reconstruction = self.slice[self.spinBox_top.value():-self.spinBox_bottom.value()-1, self.spinBox_left.value():-self.spinBox_right.value()-1]
        #original_reconstruction = self.slice
        #print('original_reconstruction.shape', original_reconstruction.shape)

        # set image dimensions only for the first time or when scan-type was changed
        if self.new == 1:
            self.pv_rec['dimension'] = [
                {'size': self.slice.shape[1], 'fullSize': self.slice.shape[1], 'binning': 1},
                {'size': self.slice.shape[0], 'fullSize': self.slice.shape[0], 'binning': 1}]
            self.new = 0

        # 16-bit integer conversion
        #self.ima3 = 65535 * (original_reconstruction + 200) / 400
        #self.ima3 = numpy.clip(self.ima3, 1, 65534)
        #self.ima3 = numpy.around(self.ima3)
        #self.ima3 = original_reconstruction
        #print('slice_show.shape',self.ima3.shape)

        # write result to pv
        self.pv_rec['value'] = ({'floatValue': self.slice.flatten()},)
        #print('Reconstructed Slice is', self.ima3.shape)

        #find and display minimum and maximum values in reconstructed slice
        #print('minimum value found: ', numpy.amin(original_reconstruction), '     maximum value found: ',numpy.amax(original_reconstruction))

        #print('reconstruction of slice is done')

        #self.check_auto_update()



#=======================================================================================================================
#no idea why we need this, but it wouldn't work without it ;-)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    main = OnTheFlyNavigator()
    main.show()
    sys.exit(app.exec_())

#end of code