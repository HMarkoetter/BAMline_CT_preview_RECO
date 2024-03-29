import pvaccess as pva
import h5py
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QStandardItem
from PyQt5.uic import loadUiType
from pathlib import Path
import numpy as np

standard_path = r'\raid-ssd\CT'
#standard_path = r'\\gfs01\g31\FB85-MeasuredData\BAMline-CT\2022\2022_03\flat_cathode\220317_1629_92_flat_cathode_____Z25_Y7400_30000eV_10x_250ms'


Ui_HDF_Browser_Window, Q_HDF_Browser_Window = loadUiType('HDF_Image_Browser.ui')  # connect to the GUI for the program


class HDF_Browser(Ui_HDF_Browser_Window, Q_HDF_Browser_Window):

    def __init__(self):
        super(HDF_Browser, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('HDF Browser')
        self.Qchannel_name.setText('BAMline:HDF_Viewer')
        # create pva type pv for reconstruction by copying metadata from the data pv, but replacing the sizes
        # This way the ADViewer (NDViewer) plugin can be also used for visualizing reconstructions.

        self.Load.clicked.connect(self.set_path)
        self.horizontalScrollBar_slice.valueChanged.connect(self.send_image)
        self.radioButton_X.toggled.connect(self.update)
        self.radioButton_Y.toggled.connect(self.update)
        self.radioButton_Z.toggled.connect(self.update)
        self.Qchannel_name.returnPressed.connect(self.updatepva)
        self.setAcceptDrops(True)

        self.pva_image_dict = {'value': ({'booleanValue': [pva.pvaccess.ScalarType.BOOLEAN], 'byteValue':
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
        self.image = pva.PvObject(self.pva_image_dict)
        self.pvaServer_HDF_Image_Browser = pva.PvaServer(self.Qchannel_name.text(), self.image)
        self.pvaServer_HDF_Image_Browser.start()
        print(self.pvaServer_HDF_Image_Browser.getRecordNames())

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        print(file_path)
        self.set_path(file_path)

    def load_clicked(self):
        # ask for hdf5-file
        path_klick = QtWidgets.QFileDialog.getOpenFileName(self, 'Select hdf5-file, please.',
                                                           standard_path)  # put NameS for multiple files

        if path_klick[0] == '':
            print('return')
            self.buttons_activate_all()
            return

        self.set_path(path_klick[0])

    def updatepva(self):
        self.pvaServer_HDF_Image_Browser.removeAllRecords()
        self.image = pva.PvObject(self.pva_image_dict)
        self.pvaServer_HDF_Image_Browser.addRecord(self.Qchannel_name.text(), self.image, None)
        print(self.pvaServer_HDF_Image_Browser.getRecordNames())
        self.pvaServer_HDF_Image_Browser.start()
        self.update()

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
            self.image['dimension'] = [
                {'size': self.dataset.shape[2], 'fullSize': self.dataset.shape[2], 'binning': 1},
                {'size': self.dataset.shape[0], 'fullSize': self.dataset.shape[0], 'binning': 1}]
            self.image['value'] = ({'ushortValue': [0, 0, 0, 0]},)
        elif self.radioButton_Y.isChecked():
            self.horizontalScrollBar_slice.setMaximum(self.dataset.shape[2] - 1)
            self.horizontalScrollBar_slice.setMinimum(0)
            self.spinBox_slice.setMaximum(self.dataset.shape[2])
            self.spinBox_slice.setValue(round(self.dataset.shape[2] / 2))
            self.image['dimension'] = [
                {'size': self.dataset.shape[1], 'fullSize': self.dataset.shape[1], 'binning': 1},
                {'size': self.dataset.shape[0], 'fullSize': self.dataset.shape[0], 'binning': 1}]
            self.image['value'] = ({'ushortValue': [0, 0, 0, 0]},)
        elif self.radioButton_Z.isChecked():
            self.horizontalScrollBar_slice.setMaximum(self.dataset.shape[0] - 1)
            self.horizontalScrollBar_slice.setMinimum(0)
            self.spinBox_slice.setMaximum(self.dataset.shape[0])
            self.spinBox_slice.setValue(round(self.dataset.shape[0] / 2))
            self.image['dimension'] = [
                {'size': self.dataset.shape[2], 'fullSize': self.dataset.shape[2], 'binning': 1},
                {'size': self.dataset.shape[1], 'fullSize': self.dataset.shape[1], 'binning': 1}]
            self.image['value'] = ({'ushortValue': [0, 0, 0, 0]},)

        self.send_image()

    def populateTree(self):
        self.treeWidget.clear()

        def recursivePopulateTree(parent_node, data):
            # If the data is a tuple, extract the first element
            if type(data[1]) == h5py.Dataset:
                dataset_item = data[1].shape
            else:
                dataset_item = ''

            tree_node = QtWidgets.QTreeWidgetItem([data[0], str(dataset_item)])
            parent_node.addChild(tree_node)
            if type(data[1]) == h5py.Group:
                for item in data[1].items():
                    recursivePopulateTree(tree_node, item)
            return tree_node

        # add root
        topnode = QtWidgets.QTreeWidgetItem([self.f.filename])
        root = self.f["/"]
        topnode.setData(0, QtCore.Qt.UserRole, root)
        self.treeWidget.addTopLevelItem(topnode)
        for item in self.f.items():
            recursivePopulateTree(topnode, item)

    def set_path(self,path_klick):
        self.path_klick = path_klick
        # grey out the buttons while program is busy
        self.buttons_deactivate_all()
        path_klick = ''


        print('path klicked: ', self.path_klick)
        # analyse and cut the path in pieces
        print('chopped path: ', *Path(self.path_klick).parts,
              sep=', ')  # faster way to chop the path (2x faster even with import)
        self.Sample.setText(self.path_klick)

        # link a volume to the hdf-file
        self.f = h5py.File(self.path_klick, 'r')

        if '/entry/data/data' in self.f:
            self.dataset = self.f['/entry/data/data']
        elif '/Volume' in self.f:
            self.dataset = self.f['/Volume']
        else:
            print('unable to find data')

        self.populateTree()

        # fichier charge
        print('raw data volume size: ', self.dataset.shape)
        print('Z', self.dataset.shape[0])
        print('X', self.dataset.shape[1])
        print('Y', self.dataset.shape[2])

        print(self.dataset.dtype)

        if self.dataset.dtype is np.dtype(float):
            print('float')
            self.typedata = 'floatValue'
        else:
            self.typedata = 'ushortValue'



        # self.Filebrowser.setText(self.h5printR(self.f))
        print('File loaded!')
        self.update()
        self.buttons_activate_all()

    def send_image(self):
        if self.radioButton_Z.isChecked():
            self.image['value'] = ({self.typedata: self.dataset[self.spinBox_slice.value(), :, :].flatten()},)
        elif self.radioButton_X.isChecked():
            self.image['value'] = ({self.typedata: self.dataset[:, self.spinBox_slice.value(), :].flatten()},)
        elif self.radioButton_Y.isChecked():
            self.image['value'] = ({self.typedata: self.dataset[:, :, self.spinBox_slice.value()].flatten()},)

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    main = HDF_Browser()
    main.show()
    sys.exit(app.exec_())
