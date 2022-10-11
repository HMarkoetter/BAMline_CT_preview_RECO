import pvaccess as pva
import h5py
from PyQt5 import QtWidgets
from PyQt5.uic import loadUiType
from pathlib import Path

channel_name = 'HDF_Viewer'
#standard_path = r'A:\BAMline-CT'
standard_path = r'\\gfs01\g31\FB85-MeasuredData\BAMline-CT\2022\2022_03\flat_cathode\220317_1629_92_flat_cathode_____Z25_Y7400_30000eV_10x_250ms'


Ui_HDF_Browser_Window, Q_HDF_Browser_Window = loadUiType('HDF_Image_Browser.ui')  # connect to the GUI for the program


class HDF_Browser(Ui_HDF_Browser_Window, Q_HDF_Browser_Window):

    def __init__(self):
        super(HDF_Browser, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('HDF Browser')
        # create pva type pv for reconstruction by copying metadata from the data pv, but replacing the sizes
        # This way the ADViewer (NDViewer) plugin can be also used for visualizing reconstructions.

        self.Load.clicked.connect(self.set_path)
        self.spinBox_slice.valueChanged.connect(self.send_image)
        self.horizontalScrollBar_slice.valueChanged.connect(self.send_image)
        self.radioButton_X.toggled.connect(self.update)
        self.radioButton_Y.toggled.connect(self.update)
        self.radioButton_Z.toggled.connect(self.update)

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

        self.image = pva.PvObject(pva_image_dict)
        self.pvaServer = pva.PvaServer(channel_name, self.image)
        self.Qchannel_name.setText(channel_name)
        self.pvaServer.start()

    def buttons_deactivate_all(self):
        self.Load.setEnabled(False)
        self.spinBox_slice.setEnabled(False)
        self.radioButton_X.setEnabled(False)
        self.radioButton_Y.setEnabled(False)
        self.radioButton_Z.setEnabled(False)
        self.horizontalScrollBar_slice.setEnabled(False)
        self.Load.setText('Busy')

    def buttons_activate_all(self):
        self.Load.setEnabled(True)
        self.spinBox_slice.setEnabled(True)
        self.radioButton_X.setEnabled(True)
        self.radioButton_Y.setEnabled(True)
        self.radioButton_Z.setEnabled(True)
        self.radioButton_Z.setChecked(True)
        self.horizontalScrollBar_slice.setEnabled(True)
        self.Load.setText('Load')

    def update(self):
        if self.radioButton_X.isChecked():
            self.horizontalScrollBar_slice.setMaximum(self.dataset.shape[1] - 1)
            self.horizontalScrollBar_slice.setMinimum(0)
            self.spinBox_slice.setMaximum(self.dataset.shape[1])
            self.spinBox_slice.setValue(round(self.dataset.shape[1] / 2))
        elif self.radioButton_Y.isChecked():
            self.horizontalScrollBar_slice.setMaximum(self.dataset.shape[2] - 1)
            self.horizontalScrollBar_slice.setMinimum(0)
            self.spinBox_slice.setMaximum(self.dataset.shape[2])
            self.spinBox_slice.setValue(round(self.dataset.shape[2] / 2))
        elif self.radioButton_Z.isChecked():
            self.horizontalScrollBar_slice.setMaximum(self.dataset.shape[0] - 1)
            self.horizontalScrollBar_slice.setMinimum(0)
            self.spinBox_slice.setMaximum(self.dataset.shape[0])
            self.spinBox_slice.setValue(round(self.dataset.shape[0] / 2))

    def set_path(self):
        # grey out the buttons while program is busy
        self.buttons_deactivate_all()
        path_klick = ''

        # ask for hdf5-file
        path_klick = QtWidgets.QFileDialog.getOpenFileName(self, 'Select hdf5-file, please.',
                                                           standard_path)  # put NameS for multiple files

        if path_klick[0] == '':
            print('return')
            self.buttons_activate_all()
            return


        self.path_klick = path_klick[0]
        print('path klicked: ', self.path_klick)
        # analyse and cut the path in pieces
        print('chopped path: ', *Path(self.path_klick).parts,
              sep=', ')  # faster way to chop the path (2x faster even with import)
        self.Sample.setText(self.path_klick)

        # link a volume to the hdf-file
        self.f = h5py.File(self.path_klick, 'r')

        # def print_attrs(name, obj):
        #     # Create indent
        #     shift = name.count('/') * '    '
        #     item_name = name.split("/")[-1]
        #     tree = shift + item_name
        #     try:
        #         if isinstance(obj, h5py.Dataset):
        #             tree += shift + str(obj.shape)
        #     except:
        #         pass
        #     return tree
        # self.structure += self.f.visititems(print_attrs)

        self.dataset = self.f['/entry/data/data']
        # fichier charge
        print('raw data volume size: ', self.dataset.shape)
        print('Z', self.dataset.shape[0])
        print('X', self.dataset.shape[1])
        print('Y', self.dataset.shape[2])

        # self.Filebrowser.setText(self.h5printR(self.f))
        print('File loaded!')
        self.spinBox_slice.setMaximum(self.dataset.shape[0])
        self.buttons_activate_all()

    def send_image(self):
        if self.radioButton_Z.isChecked():
            self.image['dimension'] = [
                {'size': self.dataset.shape[2], 'fullSize': self.dataset.shape[2], 'binning': 1},
                {'size': self.dataset.shape[1], 'fullSize': self.dataset.shape[1], 'binning': 1}]
            self.image['value'] = ({'ushortValue': self.dataset[self.spinBox_slice.value(), :, :].flatten()},)

        elif self.radioButton_X.isChecked():
            self.image['dimension'] = [
                {'size': self.dataset.shape[2], 'fullSize': self.dataset.shape[2], 'binning': 1},
                {'size': self.dataset.shape[0], 'fullSize': self.dataset.shape[0], 'binning': 1}]
            self.image['value'] = ({'ushortValue': [0,0,0,0]},)
            self.image['value'] = ({'ushortValue': self.dataset[:, self.spinBox_slice.value(), :].flatten()},)

        elif self.radioButton_Y.isChecked():
            self.image['dimension'] = [
                {'size': self.dataset.shape[1], 'fullSize': self.dataset.shape[1], 'binning': 1},
                {'size': self.dataset.shape[0], 'fullSize': self.dataset.shape[0], 'binning': 1}]
            self.image['value'] = ({'ushortValue': self.dataset[:, :, self.spinBox_slice.value()].flatten()},)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    main = HDF_Browser()
    main.show()
    sys.exit(app.exec_())
