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
import epics
import matplotlib.pyplot as plt
#import Image


# On-the-fly Navigator
version =  "Version 2023.07.31 a"

#Install ImageJ-PlugIn: EPICS AreaDetector NTNDA-Viewer, look for the channel specified here under channel_name, consider multiple users on servers!!!
channel_name = 'BAMline:Navigator'
channel_name_rec = 'BAMline:NavigatorReco'

#standard_path = "C:/temp/HDF5-Reading/220130_1734_604_J1_anode_half_cell_in-situ_Z30_Y5430_15000eV_1p44um_500ms/" # '/mnt/raid/CT/2022/'
standard_path = r'C:/delete/reg_data/18_230606_2044_AlTi_F_Ref_tomo___Z25_Y6500_25000eV_10x_400ms'

Ui_on_the_fly_Navigator_Window, Q_on_the_fly_Navigator_Window = loadUiType('on_the_fly_navigator.ui')  # connect to the GUI for the program

#plt.ion()
#fig,(ax,ax2) = plt.subplots(2,1)
#imgplotted = ax.imshow(numpy.zeros((6000,2560)),cmap='gray',vmin=0,vmax=16384)
#imgplotted = ax.imshow(numpy.zeros((720,2560)),cmap='gray',vmin=0,vmax=65535)
#angles, = ax2.plot(0,0,'o',color='red', )


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

        self.reco_rec = pva.PvObject(pva_image_dict)
        self.pvaServer_rec = pva.PvaServer(channel_name_rec, self.reco_rec)
        self.pvaServer_rec.start()

        #self.fig, self.ax = plt.subplots()

        #pvname = "PCOEdge:image1:ArrayData"
        #self.pvname = "PEGAS:miocb0101004.RBV"
        #img_pv = epics.PV(pvname, auto_monitor=True)

        self.omega_pv = epics.PV("PEGAS:miocb0102002.RBV")
        self.piezo45_pv = epics.PV("micronix:m2.RBV")
        self.piezo135_pv = epics.PV("micronix:m1.RBV")
        self.energy_pv = epics.PV("Energ:25000007rbv")
        self.distance_pv = epics.PV("faulhaber:m1.RBV")
        self.lens_pv = epics.PV("OMS58:25009007_MnuAct.SVAL")
        self.exp_time_pv = epics.PV("PCOEdge:cam1:AcquireTime_RBV")
        self.aqp_time_pv = epics.PV("PCOEdge:cam1:AcquirePeriod")
        self.W_velocity_pv = epics.PV("PEGAS:miocb0102002.VELO")

        #self.sizeX_pv = epics.PV("PCOEdge:cam1:SizeX_RBV")
        #self.sizeY_pv = epics.PV("PCOEdge:cam1:SizeY_RBV")
        #self.binningx_pv = epics.PV("PCOEdge:cam1:BinX_RBV")
        #self.binningy_pv = epics.PV("PCOEdge:cam1:BinY_RBV")

        self.sizeX_pv = epics.PV("PCOEdge:ROI1:SizeX_RBV")
        self.sizeY_pv = epics.PV("PCOEdge:ROI1:SizeY_RBV")
        self.binningx_pv = epics.PV("PCOEdge:ROI1:BinX_RBV")
        self.binningy_pv = epics.PV("PCOEdge:ROI1:BinY_RBV")

        self.binningx,self.binningy = self.binningx_pv.get(),self.binningy_pv.get()

        #self.sizeX, self.sizeY = round(self.sizeX_pv.get()/self.binningx), round(self.sizeY_pv.get()/self.binningy)
        self.sizeX, self.sizeY = self.sizeX_pv.get(),  self.sizeY_pv.get()

        self.ringbuffer_exists = 0

        if self.W_velocity_pv.get() != 0:
            print('velocity: ', self.W_velocity_pv.get(), '     exp time: ', self.exp_time_pv.get(), '      size X: ', self.sizeX_pv.get())
            self.ringbuffer_size = (
            round(360 / (self.W_velocity_pv.get() * self.exp_time_pv.get())), 1, self.sizeX_pv.get())
            print('ringbuffer_size', self.ringbuffer_size)
        else:
            print('No rotation detected!')

        self.prefill_CORs()
        self.update_pixel_size()

        self.i = 0

        self.image_pv = epics.PV("PCOEdge:image1:ArrayData", auto_monitor=True)
        #self.image_pv = epics.PV("PCOEdge:Pva1:Image:ArrayData", auto_monitor=True)
        self.image_pv.add_callback(self.update)

        self.pv_rec['dimension'] = [
            {'size': self.ringbuffer_size[2], 'fullSize': self.ringbuffer_size[2], 'binning': 1},
            {'size': int(self.ringbuffer_size[0]/2), 'fullSize': int(self.ringbuffer_size[0]/2), 'binning': 1}]

        self.ringbuffer = numpy.ones(self.ringbuffer_size, dtype='H')
        self.ringbuffer_Micos_W = numpy.zeros(self.ringbuffer_size[0], dtype=numpy.float32)

        self.ringbuffer_exists = 1
        print('ringbuffer created with size: ', self.ringbuffer.shape)

        self.sino_chopped = numpy.zeros((int(self.ringbuffer_size[0]/2),1, self.ringbuffer_size[2]), dtype='H')

        self.ruler_grid_line_thickness = 2
        self.rotation_offset = 45   #still under question
        self.label_x = 'Piezo 45 [um]'
        self.label_y = 'Piezo 135 [um]'




    def set_path(self):
        print('function set_path')

        self.new = 1
        self.extend_FOV_fixed_ImageJ_Stream = 1.0
        self.ruler_grid_line_thickness = 2
        self.rotation_offset = 45   #still under question
        self.label_x = 'Piezo 45 [um]'
        self.label_y = 'Piezo 135 [um]'

        #ask for hdf5-file
        path_klick = QtWidgets.QFileDialog.getOpenFileName(self, 'Select hdf5-file, please.', standard_path)

        if path_klick[0]:
            self.path_klick = path_klick[0]
            print('path klicked: ', self.path_klick)
            self.cut_path_name()
        else:
            print("User cancelled the dialog.")
            self.buttons_activate_load()

    def create_ringbuffer(self):
        self.ringbuffer = numpy.ones(self.ringbuffer_size, dtype='H')
        self.ringbuffer_Micos_W = numpy.zeros(self.ringbuffer_size[0], dtype=numpy.float32)

        self.ringbuffer_exists = 1
        print('ringbuffer created with size: ', self.ringbuffer.shape)

        self.sino_chopped = numpy.zeros(((self.ringbuffer_size[0]/2),1, self.ringbuffer_size[2]), dtype='H')


        #self.prefill_CORs()
        #self.update_pixel_size()

    def update(self, **kwargs):


        print('update function')
        rawimgflat = self.image_pv.get()
        #print(self.image_pv.get())

        #print('aqp_time_pv', self.aqp_time_pv.get(), 'exp_time_pv', self.exp_time_pv.get(), 'omega_pv', self.omega_pv.get(),'piezo45_pv', self.piezo45_pv.get(),'piezo135_pv', self.piezo135_pv.get(),'energy_pv', self.energy_pv.get(),'distance_pv', self.distance_pv.get(),'lens_pv', self.lens_pv.get(as_string=True),'sizeX_pv', self.sizeX_pv.get(),'sizeY_pv', self.sizeY_pv.get())
        #self.im_size = (self.sizeX,self.sizeY)
        rawimg2d = numpy.frombuffer(rawimgflat, dtype='H').reshape((self.sizeY,self.sizeX))
        print(rawimg2d.shape)

        #self.ringbuffer[self.i % self.ringbuffer_size[0],:,:] = rawimg2d[self.slice_number.value(), : ]
        self.ringbuffer[self.i % self.ringbuffer_size[0],:,:] = rawimgflat[-(round(self.sizeY / 2)) * self.sizeX: -(round(self.sizeY / 2) -1) * self.sizeX]

        self.ringbuffer_Micos_W[self.i % self.ringbuffer_size[0]] = float(self.omega_pv.get())
        print('Micos_W Ringbuffer', float(self.omega_pv.get()))
        #, self.ringbuffer_Micos_W)
        self.progressBar.setValue(self.omega_pv.get())
        if (self.i % 5) == 0:
            print('FEEDING IMAGE')
            sinogram = self.ringbuffer[:,0,:]
            #imgplotted.set_data(sinogram)
            #angles.set_data(float(self.omega_pv.get()) % 360, self.i)
            #fig.canvas.flush_events()
            #self.slice_show = sinogram.astype(numpy.float32)

            self.read_parameter()

            position = self.i % self.ringbuffer_size[0]
            print('position', position)
            if position >= int(self.ringbuffer_size[0]/2):
                self.sino_chopped[:,0,:] = sinogram[position-int(self.ringbuffer_size[0]/2):position,:]
            else:
                self.sino_chopped[-position-1:,0,:] = sinogram[:position+1,:]
                self.sino_chopped[:int(self.ringbuffer_size[0]/2)-position,0,:] = sinogram[-int(self.ringbuffer_size[0]/2) + position:,:]

            self.sino_chopped = self. sino_chopped.astype(numpy.float32)
            # write result to pv
            self.pv_rec['value'] = ({'floatValue': self.sino_chopped.flatten()},)

            #plt.imshow(sinogram, cmap='gray')
            #plt.show()
            print(int(self.ringbuffer_size[0]/2))
            self.extended_sinos = tomopy.minus_log(self.sino_chopped)
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}

            self.slice = tomopy.recon(self.extended_sinos, numpy.linspace(0,math.pi,int(self.ringbuffer_size[0]/2),endpoint=False)+((self.i % self.ringbuffer_size[0])/self.ringbuffer_size[0])*2*math.pi,
                                      center=float(self.COR),
                                      algorithm=tomopy.astra,
                                  options=options)

            if self.enable_grid.isChecked() == True:
                self.add_ruler()

            self.reco_rec['dimension'] = [
                {'size': self.slice.shape[1], 'fullSize': self.slice.shape[1], 'binning': 1},
                {'size': self.slice.shape[2], 'fullSize': self.slice.shape[2], 'binning': 1}]
            self.reco_rec['value'] = ({'floatValue': self.slice.flatten()},)

        print('i', self.i, 'Modulus:', self.i % self.ringbuffer_size[0])
        self.i = self.i +1

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
        self.f = h5py.File(self.path_klick, 'r', libver='latest', swmr=True)
        self.vol_proxy = self.f['/entry/data/data']
        #self.line_proxy = self.f['/entry/instrument/NDAttributes/CT_MICOS_W']
        self.line_proxy = self.f['/entry/instrument/NDAttributes/SAMPLE_MICOS_W2']

        print('raw data volume size: ', self.vol_proxy.shape)

        #self.prefill_parameter()
        #print('try to go to function check_180 ')
        self.check_180()    #this will end in check_auto_update



#===========================================================
    def buttons_activate_reco(self):
        print('function buttons_activate_reco')


    def prefill_parameter(self):
        print('function prefill_parameter')
        #self.prefill_slice_number()
        #self.get_rotation_angles()
        #self.find_rotation_start()
        #self.prefill_CORs()
        #self.prefill_pixel_size()
        #self.prefill_binning()
        #self.prefill_energy()
        #self.prefill_distance()


    def check_180(self):
        print('function check_180')
        while self.graph[-1] < 180:
            self.get_rotation_angles()
            print('waiting for sufficient data...', self.graph[-1])
            time.sleep(1)
        print("Enough data to proceed. angle:", self.graph[-1])
        self.check_auto_update()

    def check_auto_update(self):    #AUTO UPDATE ON/OFF?
        print('function check_auto_update')
        i=0
        while i < 1:
            j=0
            while self.auto_update.isChecked() == False:
                print('waiting for auto update...')
                QtWidgets.QApplication.processEvents()
                time.sleep(5)
            print('auto update requested')
            while self.auto_update.isChecked() == True:
                j = j + 1
                time_begin = time.time()
                self.read_parameter()
                time_read_parameter = time.time()
                #print('time_read_parameter',time_read_parameter - time_begin)
                QtWidgets.QApplication.processEvents()
                self.load_data()
                time_load_data = time.time()
                #print('time_load_data', time_load_data - time_begin)
                QtWidgets.QApplication.processEvents()
                self.reconstruct()
                time_reconstruct = time.time()
                print('iteration:', j, '  time:', round((time_reconstruct - time_begin)*1000)/1000, 'fps:', round(1/(time_reconstruct - time_begin)))
                QtWidgets.QApplication.processEvents()

    def read_parameter(self):
        #self.get_rotation_angles()
        self.update_pixel_size()
        #self.prefill_binning()
        #self.prefill_energy()
        #self.prefill_distance()


    def buttons_deactivate_all(self):
        #self.pushLoad.setEnabled(False)
        #self.slice_number.setEnabled(False)
        self.COR_1.setEnabled(False)
        self.COR_2.setEnabled(False)
        self.COR_3.setEnabled(False)
        self.COR_4.setEnabled(False)

    def buttons_activate_load(self):
        self.pushLoad.setEnabled(True)


    def prefill_slice_number(self):  #get the image dimensions and prefill slice number
        self.slice_number.setMaximum(self.vol_proxy.shape[1]-1)
        self.slice_number.setMinimum(0)
        time.sleep(1)
        self.slice_number.setValue(round(self.vol_proxy.shape[1]/2))    # be careful with an infinite loop when setValue actually triggers valueChanged. Therefore, auto_update starts off
        print('Function prefill_slice_number: Set middle height as slice number:  ', self.slice_number.value())
        self.slice_number.setEnabled(True)

    def get_rotation_angles(self):
        #self.f = h5py.File(self.path_klick, 'r',libver='latest', swmr=True)
        #self.line_proxy = self.f['/entry/instrument/NDAttributes/CT_MICOS_W']
        #self.line_proxy = self.f['/entry/instrument/NDAttributes/SAMPLE_MICOS_W2']
        self.line_proxy.id.refresh()
        self.graph = numpy.array(self.line_proxy)
        print('Function get_rotation_angles: Found number of angles:  ', self.graph.shape[0], '      current angle: ', self.graph[-1])


    def prefill_CORs(self):
        if self.COR_1.value() == 0:
            print('preset CORs to half image size')
            self.COR_1.setValue(round(self.sizeX_pv.get() / 2))
            self.COR_2.setValue(round(self.sizeX_pv.get() / 2))
            self.COR_3.setValue(round(self.sizeX_pv.get() / 2))
            self.COR_4.setValue(round(self.sizeX_pv.get() / 2))

    def prefill_pixel_size(self):

        if '/entry/instrument/NDAttributes/CT_Pixelsize' in self.f:
            self.pixel_proxy = self.f['/entry/instrument/NDAttributes/CT_Pixelsize']
            self.pixel_size.setValue(self.pixel_proxy[-1])
            #print('Function prefill_pixel_size: ', self.pixel_proxy[-1])
        else:
            self.pixel_size.setValue(1)

    def update_pixel_size(self):

        if self.lens_pv.get() == '2x':
            self.pixel_size_set = 3.6
            self.buttons_deactivate_all()
            self.COR_1.setEnabled(True)
            self.COR = self.COR_1.value()
            self.spinBox_ruler_grid = self.spinBox_ruler_grid_1.value()
        elif self.lens_pv.get() == '5x':
            self.pixel_size_set = 1.44
            self.buttons_deactivate_all()
            self.COR_2.setEnabled(True)
            self.COR = self.COR_2.value()
            self.spinBox_ruler_grid = self.spinBox_ruler_grid_2.value()
        elif self.lens_pv.get() == '10x':
            self.pixel_size_set = 0.72
            self.buttons_deactivate_all()
            self.COR_3.setEnabled(True)
            self.COR = self.COR_3.value()
            self.spinBox_ruler_grid = self.spinBox_ruler_grid_3.value()
        elif self.lens_pv.get() == '20x':
            self.pixel_size_set = 0.36
            self.buttons_deactivate_all()
            self.COR_4.setEnabled(True)
            self.COR = self.COR_4.value()
            self.spinBox_ruler_grid = self.spinBox_ruler_grid_4.value()
        else:
            self.buttons_deactivate_all()

        print('pixel size: ', self.pixel_size_set)
        self.pixel_size.setValue(self.pixel_size_set)


    def prefill_binning(self):
        if '/entry/instrument/NDAttributes/Binning_X' in self.f:
            self.binning_proxy = self.f['/entry/instrument/NDAttributes/Binning_X']
            self.binning.setValue(self.binning_proxy[-1])
            print('Function prefill_binning: ', self.binning_proxy[-1])
        else:
            self.binning.setValue(1)
            print('Function prefill_binning: Not found. Set to 1')





    def prefill_energy(self):
        #self.energy_pv.get()
        self.doubleSpinBox_Energy_2.setValue(round(float(self.energy_pv.get()) * 100) / 100)
        """
        if '/entry/instrument/NDAttributes/DMM_Energy' in self.f:
            self.energy_proxy = self.f['/entry/instrument/NDAttributes/DMM_Energy']
            self.doubleSpinBox_Energy_2.setValue(round(self.energy_proxy[-1]*100)/100)
            #print('Function prefill_energy:', round(self.energy_proxy[-1]*100)/100)
        else:
            self.doubleSpinBox_Energy_2.setValue(1)
            #print('Function prefill_energy: Energy not found')
        """

    def prefill_distance(self):

        self.doubleSpinBox_distance_2.setValue(round(float(self.distance_pv.get()) + 25))
        """
        if '/entry/instrument/NDAttributes/CT-Kamera-Z' in self.f:
            self.distance_proxy = self.f['/entry/instrument/NDAttributes/CT-Kamera-Z']
            self.doubleSpinBox_distance_2.setValue(round(self.distance_proxy[-1] + 25))
            #print('Function prefill_distance:', round(self.distance_proxy[-1] + 25))
        else:
            self.doubleSpinBox_distance_2.setValue(0)
            #print('Function prefill_distance: Not found. Set to 0')
        """
        QtWidgets.QApplication.processEvents()


    def load_data(self):
        self.f = h5py.File(self.path_klick, 'r', libver='latest', swmr=True)
        self.w = self.graph

        #prefill rotation-speed[°/img]        #Polynom fit for the angles
        poly_coeff = numpy.polyfit(numpy.arange(len(self.w[round((self.w.shape[0] + 1) /4) : round((self.w.shape[0] + 1) * 3/4) ])), self.w[round((self.w.shape[0] + 1) /4) : round((self.w.shape[0] + 1) * 3/4) ], 1, rcond=None, full=False, w=None, cov=False)
        self.speed_W = poly_coeff[0]
        self.number_of_used_projections = round(180 / self.speed_W)

        # load recent 180deg sino
        self.vol_proxy = self.f['/entry/data/data']
        Sino = self.vol_proxy[- self.number_of_used_projections : , self.slice_number.value(), :]
        self.Norm = Sino


    def add_ruler(self):

        if self.piezo45_pv.get(as_string=True) == 'None':
            self.piezo_45_proxy = 0
        else:
            self.piezo_45_proxy = self.piezo45_pv.get()
        print('piezo 45', self.piezo_45_proxy)

        if self.piezo135_pv.get(as_string=True) == 'None':
            self.piezo_135_proxy = 0
        else:
            self.piezo_135_proxy = self.piezo135_pv.get()
        print('piezo 135', self.piezo_135_proxy)

        self.ruler_grid_color = math.ceil(numpy.max(self.slice))

        # draws a circle with the detector size as diameter
        cv2.circle(self.slice, (round(self.slice.shape[1] / 2), round(self.slice.shape[1] / 2)), round(self.slice.shape[1]/2), self.ruler_grid_color, self.ruler_grid_line_thickness)

        if self.grid_micrometer.isChecked() == True:

            # add label x
            cv2.putText(self.slice, self.label_x, (
            round(self.slice.shape[1] * self.pixel_size.value() / 2) + 20, round(self.ruler_grid_line_thickness) * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness / 2), self.ruler_grid_color,
                        thickness=self.ruler_grid_line_thickness)

            # add ruler +X
            for r in range(round(self.slice.shape[1] * self.pixel_size.value() * self.binning.value() / 2),
                           round(self.slice.shape[1] * self.pixel_size.value() * self.binning.value()),
                           round(self.spinBox_ruler_grid)):
                cv2.line(self.slice, (round(r / (self.binning.value() * self.pixel_size.value())), 0),
                         (round(r / (self.pixel_size.value() * self.binning.value())), self.slice.shape[0]), self.ruler_grid_color, self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round(1000 * self.piezo_45_proxy/  5) * 5   +   round( r / 5) * 5   -   round(self.slice.shape[1] * (self.pixel_size.value() * self.binning.value() / 10)) * 5),
                            (round(r / (self.pixel_size.value() * self.binning.value())) + 20, round(self.ruler_grid_line_thickness)*20), cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness/2),
                            self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)

            # add ruler -X
            for r in range(round(self.slice.shape[1] * self.pixel_size.value() * self.binning.value() / 2), 0,
                           -round(self.spinBox_ruler_grid)):
                cv2.line(self.slice, (round(r / (self.pixel_size.value() * self.binning.value())), 0),
                         (round(r / (self.pixel_size.value() * self.binning.value())), self.slice.shape[0]), self.ruler_grid_color, self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round(1000 * self.piezo_45_proxy / 5) * 5  +  round(
                    r  / 5) * 5  -  round(self.slice.shape[1] * (self.pixel_size.value() * self.binning.value() / 10)) * 5),
                            (round(r / (self.pixel_size.value() * self.binning.value() )) + 20, round(self.ruler_grid_line_thickness)*20), cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness/2),
                            self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)


            # add label Y
            cv2.putText(self.slice, self.label_y, (20, round(self.slice.shape[1] * self.pixel_size.value() / 2) -40),
                        cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness / 2), self.ruler_grid_color,
                        thickness=self.ruler_grid_line_thickness)

            # add ruler +Y
            for r in range(round(self.slice.shape[1] * self.pixel_size.value() * self.binning.value() / 2),
                           round(self.slice.shape[1] * self.pixel_size.value()* self.binning.value()),
                           round(self.spinBox_ruler_grid)):
                cv2.line(self.slice, (0, round(r / (self.pixel_size.value()* self.binning.value()))),
                         (self.slice.shape[1], round(r / (self.pixel_size.value()* self.binning.value()))), self.ruler_grid_color, self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round(1000 * self.piezo_135_proxy / 5) * 5  +  round(
                    r / 5) * 5 - round(self.slice.shape[1] * (self.pixel_size.value() * self.binning.value() / 10)) * 5),
                            (20, round(r / (self.pixel_size.value() * self.binning.value())) + round(self.ruler_grid_line_thickness)*20), cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness/2),
                            self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)

            # add ruler -Y
            for r in range(round(self.slice.shape[1] * self.pixel_size.value() * self.binning.value()/ 2), 0,
                           -round(self.spinBox_ruler_grid)):
                cv2.line(self.slice, (0, round(r / (self.pixel_size.value()* self.binning.value()))),
                         (self.slice.shape[1], round(r / (self.pixel_size.value()* self.binning.value()))), self.ruler_grid_color, self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round(1000 * self.piezo_135_proxy / 5) * 5 + round(
                    r / 5) * 5 - round(self.slice.shape[1] * (self.pixel_size.value() * self.binning.value() / 10)) * 5),
                            (20, round(r / (self.pixel_size.value()* self.binning.value())) + round(self.ruler_grid_line_thickness)*20), cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness/2),
                            self.ruler_grid_color, thickness=self.ruler_grid_line_thickness)


        if self.grid_pixel.isChecked() == True:

            # add label x
            cv2.putText(self.slice, 'Pixel', (
                round(self.slice.shape[1] * self.pixel_size.value() / 2) + 20,
                round(self.ruler_grid_line_thickness) * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness / 2), self.ruler_grid_color,
                        thickness=self.ruler_grid_line_thickness)

            # add ruler +X
            for r in range(round(self.slice.shape[1] / 2), self.slice.shape[1], round(self.spinBox_pixel_grid.value())):
                cv2.line(self.slice, (r, 0),   (r, self.slice.shape[1]), (65535, 65535, 65535), self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (r + 20, round(self.ruler_grid_line_thickness/2)*20), cv2.FONT_HERSHEY_SIMPLEX, self.ruler_grid_line_thickness/2, (65535, 65535, 65535), thickness=self.ruler_grid_line_thickness)

            # add ruler -X
            for r in range(round(self.slice.shape[1] / 2), 0, -round(self.spinBox_pixel_grid.value())):
                #print(r)
                cv2.line(self.slice, (r, 0),   (r, self.slice.shape[1]), (65535, 65535, 65535), self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (r + 20, round(self.ruler_grid_line_thickness/2)*20), cv2.FONT_HERSHEY_SIMPLEX, self.ruler_grid_line_thickness/2, (65535, 65535, 65535), thickness=self.ruler_grid_line_thickness)


            # add label Y
            cv2.putText(self.slice, 'Pixel',
                        (20, round(self.slice.shape[1] * self.pixel_size.value() / 2) - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, (self.ruler_grid_line_thickness / 2), self.ruler_grid_color,
                        thickness=self.ruler_grid_line_thickness)

            # add ruler +Y
            for r in range(round(self.slice.shape[1] / 2), self.slice.shape[1], round(self.spinBox_pixel_grid.value())):
                cv2.line(self.slice, (0, r), (self.slice.shape[1], r), (65535, 65535, 65535), self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (20,r + round(self.ruler_grid_line_thickness/2)*20), cv2.FONT_HERSHEY_SIMPLEX, self.ruler_grid_line_thickness/2, (65535, 65535, 65535), thickness=self.ruler_grid_line_thickness)

            # add ruler -Y
            for r in range(round(self.slice.shape[1] / 2), 0, -round(self.spinBox_pixel_grid.value())):
                cv2.line(self.slice, (0, r), (self.slice.shape[1], r), (65535, 65535, 65535), self.ruler_grid_line_thickness)
                cv2.putText(self.slice, str(round((r - (self.slice.shape[1] / 2)) / 5) * 5), (20,r + round(self.ruler_grid_line_thickness/2)*20), cv2.FONT_HERSHEY_SIMPLEX, self.ruler_grid_line_thickness/2, (65535, 65535, 65535), thickness=self.ruler_grid_line_thickness)


    def reconstruct(self):
        QtWidgets.QApplication.processEvents()

        self.full_size = self.Norm.shape[1]
        self.number_of_projections = self.Norm.shape[0]

        self.number_of_used_projections = round(180 / self.speed_W)

        # create list with all projection angles
        new_list = (numpy.arange(self.number_of_used_projections) * self.speed_W + self.graph[-1] + self.rotation_offset) * math.pi / 180



        if self.pixel_size.value() != 1:
            if self.pixel_proxy[-1] == 3.61:
                self.COR = self.COR_1.value()
            elif self.pixel_proxy[-1] == 1.44:
                self.COR = self.COR_2.value()
            elif self.pixel_proxy[-1] == 0.72:
                self.COR = self.COR_3.value()
            elif self.pixel_proxy[-1] == 0.36:
                self.COR = self.COR_4.value()
            else:
                self.COR = 50
        else:
            self.COR = self.COR_1.value()
            print('Pixel Size unknown. Use COR_1')

        center_list = [self.COR + round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size)] * (self.number_of_used_projections)

        # create one sinogram in the form [z, y, x]
        transposed_sinos = numpy.zeros((self.number_of_used_projections, 1, self.full_size), dtype=float)
        transposed_sinos[:,0,:] = self.Norm[-self.number_of_used_projections : , : ]

        #extend data with calculated parameter, compute logarithm, remove NaN-values
        log_sinos = tomopy.minus_log(transposed_sinos)
        log_sinos = numpy.nan_to_num(log_sinos, copy=True, nan=1.0, posinf=1.0, neginf=1.0)
        extended_sinos = tomopy.misc.morph.pad(log_sinos, axis=2,
                                               npad=round(self.extend_FOV_fixed_ImageJ_Stream * self.full_size),
                                               mode='edge')

        #reconstruct one slice

        if self.GPU_CUDA.isChecked() == True:
            options = {'proj_type': 'cuda', 'method': 'FBP_CUDA'}
            slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm=tomopy.astra, options=options)
        else:
            slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm='gridrec', filter_name='shepp')


        #slices = tomopy.recon(extended_sinos, new_list, center=center_list, algorithm='gridrec', filter_name='shepp')
        slices = slices[:,round(self.full_size/4):-round(self.full_size/4),round(self.full_size/4):-round(self.full_size/4)]
        slices = tomopy.circ_mask(slices, axis=0, ratio=1.0)
        slices = slices * (10000 / self.pixel_size.value())
        self.slice = slices[0,:,:]   #reduce dimensions from 3 to 2

        self.add_ruler()

        # set image dimensions only for the first time or when scan-type was changed
        if self.new == 1:
            self.pv_rec['dimension'] = [
                {'size': self.slice.shape[1], 'fullSize': self.slice.shape[1], 'binning': 1},
                {'size': self.slice.shape[0], 'fullSize': self.slice.shape[0], 'binning': 1}]
            self.new = 0

        # write result to pv
        self.pv_rec['value'] = ({'floatValue': self.slice.flatten()},)


#=======================================================================================================================
#no idea why we need this, but it wouldn't work without it ;-)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    main = OnTheFlyNavigator()
    main.show()
    sys.exit(app.exec_())

#end of code