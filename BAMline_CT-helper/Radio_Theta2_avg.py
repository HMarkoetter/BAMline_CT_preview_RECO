#Install ImageJ-PlugIn: EPICS AreaDetector NTNDA-Viewer, look for the channel specified here under channel_name, consider multiple users on servers!!!
channel_name = 'BAMline:Radio_Theta2_avg'
standard_path = "/raid/CT/2023/2023_03/Markoetter/daisy_lam3_top_mystery/" # '/mnt/raid/CT/2022/'

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
import scipy
import pvaccess as pva      #to install search for "pvapy"


Ui_Theta2_avg_Window, Q_Theta2_avg_Window = loadUiType('Radio_Theta2_avg.ui')  # connect to the GUI for the program

class Radio_Theta2_avg(Ui_Theta2_avg_Window, Q_Theta2_avg_Window):


    def __init__(self):
        super(Radio_Theta2_avg, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Radio_Theta2_avg')

        #connect buttons to actions
        self.pushLoad.clicked.connect(self.set_path)
        self.pushCompute.clicked.connect(self.compute)
        self.doubleSpinBox_Shift.valueChanged.connect(self.check)


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

        self.new = 1

        # ask for hdf5-file #proj
        path_klick_proj = QtWidgets.QFileDialog.getOpenFileName(self, 'Select projections hdf5-file, please.', standard_path)
        self.path_klick_proj = path_klick_proj[0]
        print('path klicked: ', self.path_klick_proj)

        # ask for hdf5-file #ff
        path_klick_ff = QtWidgets.QFileDialog.getOpenFileName(self, 'Select flat-field hdf5-file, please.', standard_path)
        self.path_klick_ff = path_klick_ff[0]
        print('path klicked: ', self.path_klick_ff)


        # link a volume to the proj hdf-file
        f_proj = h5py.File(self.path_klick_proj, 'r')
        self.vol_proxy_proj = f_proj['/entry/data/data']
        print('raw data proj volume size: ', self.vol_proxy_proj.shape)

        # link a volume to the ff hdf-file
        f_ff = h5py.File(self.path_klick_ff, 'r')
        self.vol_proxy_ff = f_ff['/entry/data/data']
        print('raw data ff volume size: ', self.vol_proxy_ff.shape)


        self.load()


    def load(self):

        FFs = self.vol_proxy_ff
        Proj = self.vol_proxy_proj

        self.min_size = min(FFs.shape[0],Proj.shape[0])

        self.Norm_stack = numpy.divide(numpy.subtract(Proj[0:self.min_size,:,:], self.spinBox_DF.value()), numpy.subtract(FFs[0:self.min_size,:,:], self.spinBox_DF.value()))






    def check(self):
        if self.checkBox_auto.isChecked():
            self.compute()

    def compute(self):
        self.pushCompute.setText('Busy')
        self.pushLoad.setEnabled(False)
        self.pushCompute.setEnabled(False)  # DOES NOT SEEM TO WORK!

        self.Norm_stack2 = self.Norm_stack.copy()

        i = 0
        while i < self.min_size:
            print(i, ' of ',self.min_size)
            self.Norm_stack2[i,:,:] = scipy.ndimage.shift(self.Norm_stack[i,:,:], (i * self.doubleSpinBox_Shift.value(),0), order=0, mode='nearest', prefilter=True)
            i=i+1

        self.avg = numpy.mean(self.Norm_stack2, axis=0, dtype=numpy.float32)
        print(self.avg.shape)


        self.pv_rec['dimension'] = [
            {'size': self.avg.shape[1], 'fullSize': self.avg.shape[1], 'binning': 1},
            {'size': self.avg.shape[0], 'fullSize': self.avg.shape[0], 'binning': 1}]
        # write result to pv
        self.pv_rec['value'] = ({'floatValue': self.avg.flatten()},)

        self.pushCompute.setText('Compute')
        self.pushLoad.setEnabled(True)
        self.pushCompute.setEnabled(True)


#no idea why we need this, but it wouldn't work without it ;-)
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    main = Radio_Theta2_avg()
    main.show()
    sys.exit(app.exec_())

#end of code